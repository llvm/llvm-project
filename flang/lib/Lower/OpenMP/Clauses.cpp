//===-- Clauses.cpp -- OpenMP clause handling -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Clauses.h"

#include "flang/Common/idioms.h"
#include "flang/Evaluate/expression.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"

#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include <list>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>

namespace detail {
template <typename C>
llvm::omp::Clause getClauseIdForClass(C &&) {
  using namespace Fortran;
  using A = llvm::remove_cvref_t<C>; // A is referenced in OMP.inc
  // The code included below contains a sequence of checks like the following
  // for each OpenMP clause
  //   if constexpr (std::is_same_v<A, parser::OmpClause::AcqRel>)
  //     return llvm::omp::Clause::OMPC_acq_rel;
  //   [...]
#define GEN_FLANG_CLAUSE_PARSER_KIND_MAP
#include "llvm/Frontend/OpenMP/OMP.inc"
}
} // namespace detail

static llvm::omp::Clause getClauseId(const Fortran::parser::OmpClause &clause) {
  return Fortran::common::visit(
      [](auto &&s) { return detail::getClauseIdForClass(s); }, clause.u);
}

namespace Fortran::lower::omp {
using SymbolWithDesignator = std::tuple<semantics::Symbol *, MaybeExpr>;

struct SymbolAndDesignatorExtractor {
  template <typename T>
  static T &&AsRvalueRef(T &&t) {
    return std::move(t);
  }
  template <typename T>
  static T AsRvalueRef(const T &t) {
    return t;
  }

  static semantics::Symbol *symbol_addr(const evaluate::SymbolRef &ref) {
    // Symbols cannot be created after semantic checks, so all symbol
    // pointers that are non-null must point to one of those pre-existing
    // objects. Throughout the code, symbols are often pointed to by
    // non-const pointers, so there is no harm in casting the constness
    // away.
    return const_cast<semantics::Symbol *>(&ref.get());
  }

  template <typename T>
  static SymbolWithDesignator visit(T &&) {
    // Use this to see missing overloads:
    // llvm::errs() << "NULL: " << __PRETTY_FUNCTION__ << '\n';
    return SymbolWithDesignator{};
  }

  template <typename T>
  static SymbolWithDesignator visit(const evaluate::Designator<T> &e) {
    return std::make_tuple(symbol_addr(*e.GetLastSymbol()),
                           evaluate::AsGenericExpr(AsRvalueRef(e)));
  }

  static SymbolWithDesignator visit(const evaluate::ProcedureDesignator &e) {
    return std::make_tuple(symbol_addr(*e.GetSymbol()), std::nullopt);
  }

  template <typename T>
  static SymbolWithDesignator visit(const evaluate::Expr<T> &e) {
    return Fortran::common::visit([](auto &&s) { return visit(s); }, e.u);
  }

  static void verify(const SymbolWithDesignator &sd) {
    const semantics::Symbol *symbol = std::get<0>(sd);
    assert(symbol && "Expecting symbol");
    auto &maybeDsg = std::get<1>(sd);
    if (!maybeDsg)
      return; // Symbol with no designator -> OK
    std::optional<evaluate::DataRef> maybeRef =
        evaluate::ExtractDataRef(*maybeDsg);
    if (maybeRef) {
      if (&maybeRef->GetLastSymbol() == symbol)
        return; // Symbol with a designator for it -> OK
      llvm_unreachable("Expecting designator for given symbol");
    } else {
      // This could still be a Substring or ComplexPart, but at least Substring
      // is not allowed in OpenMP.
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      maybeDsg->dump();
#endif
      llvm_unreachable("Expecting DataRef designator");
    }
  }
};

SymbolWithDesignator getSymbolAndDesignator(const MaybeExpr &expr) {
  if (!expr)
    return SymbolWithDesignator{};
  return Fortran::common::visit(
      [](auto &&s) { return SymbolAndDesignatorExtractor::visit(s); }, expr->u);
}

Object makeObject(const parser::Name &name,
                  semantics::SemanticsContext &semaCtx) {
  assert(name.symbol && "Expecting Symbol");
  return Object{name.symbol, std::nullopt};
}

Object makeObject(const parser::Designator &dsg,
                  semantics::SemanticsContext &semaCtx) {
  evaluate::ExpressionAnalyzer ea{semaCtx};
  SymbolWithDesignator sd = getSymbolAndDesignator(ea.Analyze(dsg));
  SymbolAndDesignatorExtractor::verify(sd);
  return Object{std::get<0>(sd), std::move(std::get<1>(sd))};
}

Object makeObject(const parser::StructureComponent &comp,
                  semantics::SemanticsContext &semaCtx) {
  evaluate::ExpressionAnalyzer ea{semaCtx};
  SymbolWithDesignator sd = getSymbolAndDesignator(ea.Analyze(comp));
  SymbolAndDesignatorExtractor::verify(sd);
  return Object{std::get<0>(sd), std::move(std::get<1>(sd))};
}

Object makeObject(const parser::OmpObject &object,
                  semantics::SemanticsContext &semaCtx) {
  // If object is a common block, expression analyzer won't be able to
  // do anything.
  if (const auto *name = std::get_if<parser::Name>(&object.u)) {
    assert(name->symbol && "Expecting Symbol");
    return Object{name->symbol, std::nullopt};
  }
  // OmpObject is std::variant<Designator, /*common block*/ Name>;
  return makeObject(std::get<parser::Designator>(object.u), semaCtx);
}

std::optional<Object> getBaseObject(const Object &object,
                                    semantics::SemanticsContext &semaCtx) {
  // If it's just the symbol, then there is no base.
  if (!object.ref())
    return std::nullopt;

  auto maybeRef = evaluate::ExtractDataRef(*object.ref());
  if (!maybeRef)
    return std::nullopt;

  evaluate::DataRef ref = *maybeRef;

  if (std::get_if<evaluate::SymbolRef>(&ref.u)) {
    return std::nullopt;
  } else if (auto *comp = std::get_if<evaluate::Component>(&ref.u)) {
    const evaluate::DataRef &base = comp->base();
    return Object{
        SymbolAndDesignatorExtractor::symbol_addr(base.GetLastSymbol()),
        evaluate::AsGenericExpr(
            SymbolAndDesignatorExtractor::AsRvalueRef(base))};
  } else if (auto *arr = std::get_if<evaluate::ArrayRef>(&ref.u)) {
    const evaluate::NamedEntity &base = arr->base();
    evaluate::ExpressionAnalyzer ea{semaCtx};
    if (auto *comp = base.UnwrapComponent()) {
      return Object{SymbolAndDesignatorExtractor::symbol_addr(comp->symbol()),
                    ea.Designate(evaluate::DataRef{
                        SymbolAndDesignatorExtractor::AsRvalueRef(*comp)})};
    } else if (base.UnwrapSymbolRef()) {
      return std::nullopt;
    }
  } else {
    assert(std::holds_alternative<evaluate::CoarrayRef>(ref.u) &&
           "Unexpected variant alternative");
    llvm_unreachable("Coarray reference not supported at the moment");
  }
  return std::nullopt;
}

// Helper macros
#define MAKE_EMPTY_CLASS(cls, from_cls)                                        \
  cls make(const parser::OmpClause::from_cls &,                                \
           semantics::SemanticsContext &) {                                    \
    static_assert(cls::EmptyTrait::value);                                     \
    return cls{};                                                              \
  }                                                                            \
  [[maybe_unused]] extern int xyzzy_semicolon_absorber

#define MAKE_INCOMPLETE_CLASS(cls, from_cls)                                   \
  cls make(const parser::OmpClause::from_cls &,                                \
           semantics::SemanticsContext &) {                                    \
    static_assert(cls::IncompleteTrait::value);                                \
    return cls{};                                                              \
  }                                                                            \
  [[maybe_unused]] extern int xyzzy_semicolon_absorber

#define MS(x, y) CLAUSET_SCOPED_ENUM_MEMBER_CONVERT(x, y)
#define MU(x, y) CLAUSET_UNSCOPED_ENUM_MEMBER_CONVERT(x, y)

namespace clause {
MAKE_EMPTY_CLASS(AcqRel, AcqRel);
MAKE_EMPTY_CLASS(Acquire, Acquire);
MAKE_EMPTY_CLASS(Capture, Capture);
MAKE_EMPTY_CLASS(Compare, Compare);
MAKE_EMPTY_CLASS(DynamicAllocators, DynamicAllocators);
MAKE_EMPTY_CLASS(Full, Full);
MAKE_EMPTY_CLASS(Inbranch, Inbranch);
MAKE_EMPTY_CLASS(Mergeable, Mergeable);
MAKE_EMPTY_CLASS(Nogroup, Nogroup);
// MAKE_EMPTY_CLASS(NoOpenmp, );         // missing-in-parser
// MAKE_EMPTY_CLASS(NoOpenmpRoutines, ); // missing-in-parser
// MAKE_EMPTY_CLASS(NoParallelism, );    // missing-in-parser
MAKE_EMPTY_CLASS(Notinbranch, Notinbranch);
MAKE_EMPTY_CLASS(Nowait, Nowait);
MAKE_EMPTY_CLASS(OmpxAttribute, OmpxAttribute);
MAKE_EMPTY_CLASS(OmpxBare, OmpxBare);
MAKE_EMPTY_CLASS(Read, Read);
MAKE_EMPTY_CLASS(Relaxed, Relaxed);
MAKE_EMPTY_CLASS(Release, Release);
MAKE_EMPTY_CLASS(ReverseOffload, ReverseOffload);
MAKE_EMPTY_CLASS(SeqCst, SeqCst);
MAKE_EMPTY_CLASS(Simd, Simd);
MAKE_EMPTY_CLASS(Threads, Threads);
MAKE_EMPTY_CLASS(UnifiedAddress, UnifiedAddress);
MAKE_EMPTY_CLASS(UnifiedSharedMemory, UnifiedSharedMemory);
MAKE_EMPTY_CLASS(Unknown, Unknown);
MAKE_EMPTY_CLASS(Untied, Untied);
MAKE_EMPTY_CLASS(Weak, Weak);
MAKE_EMPTY_CLASS(Write, Write);

// Artificial clauses
MAKE_EMPTY_CLASS(CancellationConstructType, CancellationConstructType);
MAKE_EMPTY_CLASS(Depobj, Depobj);
MAKE_EMPTY_CLASS(Flush, Flush);
MAKE_EMPTY_CLASS(MemoryOrder, MemoryOrder);
MAKE_EMPTY_CLASS(Threadprivate, Threadprivate);

MAKE_INCOMPLETE_CLASS(AdjustArgs, AdjustArgs);
MAKE_INCOMPLETE_CLASS(AppendArgs, AppendArgs);
MAKE_INCOMPLETE_CLASS(Match, Match);
// MAKE_INCOMPLETE_CLASS(Otherwise, );   // missing-in-parser
MAKE_INCOMPLETE_CLASS(When, When);

DefinedOperator makeDefinedOperator(const parser::DefinedOperator &inp,
                                    semantics::SemanticsContext &semaCtx) {
  CLAUSET_ENUM_CONVERT( //
      convert, parser::DefinedOperator::IntrinsicOperator,
      DefinedOperator::IntrinsicOperator,
      // clang-format off
      MS(Add,      Add)
      MS(AND,      AND)
      MS(Concat,   Concat)
      MS(Divide,   Divide)
      MS(EQ,       EQ)
      MS(EQV,      EQV)
      MS(GE,       GE)
      MS(GT,       GT)
      MS(NOT,      NOT)
      MS(LE,       LE)
      MS(LT,       LT)
      MS(Multiply, Multiply)
      MS(NE,       NE)
      MS(NEQV,     NEQV)
      MS(OR,       OR)
      MS(Power,    Power)
      MS(Subtract, Subtract)
      // clang-format on
  );

  return Fortran::common::visit(
      common::visitors{
          [&](const parser::DefinedOpName &s) {
            return DefinedOperator{
                DefinedOperator::DefinedOpName{makeObject(s.v, semaCtx)}};
          },
          [&](const parser::DefinedOperator::IntrinsicOperator &s) {
            return DefinedOperator{convert(s)};
          },
      },
      inp.u);
}

ProcedureDesignator
makeProcedureDesignator(const parser::ProcedureDesignator &inp,
                        semantics::SemanticsContext &semaCtx) {
  return ProcedureDesignator{Fortran::common::visit(
      common::visitors{
          [&](const parser::Name &t) { return makeObject(t, semaCtx); },
          [&](const parser::ProcComponentRef &t) {
            return makeObject(t.v.thing, semaCtx);
          },
      },
      inp.u)};
}

ReductionOperator makeReductionOperator(const parser::OmpReductionOperator &inp,
                                        semantics::SemanticsContext &semaCtx) {
  return Fortran::common::visit(
      common::visitors{
          [&](const parser::DefinedOperator &s) {
            return ReductionOperator{makeDefinedOperator(s, semaCtx)};
          },
          [&](const parser::ProcedureDesignator &s) {
            return ReductionOperator{makeProcedureDesignator(s, semaCtx)};
          },
      },
      inp.u);
}

// --------------------------------------------------------------------
// Actual clauses. Each T (where tomp::T exists in ClauseT) has its "make".

// Absent: missing-in-parser
// AcqRel: empty
// Acquire: empty
// AdjustArgs: incomplete

Affinity make(const parser::OmpClause::Affinity &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: affinity");
}

Align make(const parser::OmpClause::Align &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: align");
}

Aligned make(const parser::OmpClause::Aligned &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpAlignedClause
  auto &t0 = std::get<parser::OmpObjectList>(inp.v.t);
  auto &t1 = std::get<std::optional<parser::ScalarIntConstantExpr>>(inp.v.t);

  return Aligned{{
      /*Alignment=*/maybeApply(makeExprFn(semaCtx), t1),
      /*List=*/makeObjects(t0, semaCtx),
  }};
}

Allocate make(const parser::OmpClause::Allocate &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpAllocateClause
  using wrapped = parser::OmpAllocateClause;
  auto &t0 = std::get<std::optional<wrapped::AllocateModifier>>(inp.v.t);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);

  if (!t0) {
    return Allocate{{/*AllocatorSimpleModifier=*/std::nullopt,
                     /*AllocatorComplexModifier=*/std::nullopt,
                     /*AlignModifier=*/std::nullopt,
                     /*List=*/makeObjects(t1, semaCtx)}};
  }

  using Tuple = decltype(Allocate::t);

  return Allocate{Fortran::common::visit(
      common::visitors{
          // simple-modifier
          [&](const wrapped::AllocateModifier::Allocator &v) -> Tuple {
            return {/*AllocatorSimpleModifier=*/makeExpr(v.v, semaCtx),
                    /*AllocatorComplexModifier=*/std::nullopt,
                    /*AlignModifier=*/std::nullopt,
                    /*List=*/makeObjects(t1, semaCtx)};
          },
          // complex-modifier + align-modifier
          [&](const wrapped::AllocateModifier::ComplexModifier &v) -> Tuple {
            auto &s0 = std::get<wrapped::AllocateModifier::Allocator>(v.t);
            auto &s1 = std::get<wrapped::AllocateModifier::Align>(v.t);
            return {
                /*AllocatorSimpleModifier=*/std::nullopt,
                /*AllocatorComplexModifier=*/Allocator{makeExpr(s0.v, semaCtx)},
                /*AlignModifier=*/Align{makeExpr(s1.v, semaCtx)},
                /*List=*/makeObjects(t1, semaCtx)};
          },
          // align-modifier
          [&](const wrapped::AllocateModifier::Align &v) -> Tuple {
            return {/*AllocatorSimpleModifier=*/std::nullopt,
                    /*AllocatorComplexModifier=*/std::nullopt,
                    /*AlignModifier=*/Align{makeExpr(v.v, semaCtx)},
                    /*List=*/makeObjects(t1, semaCtx)};
          },
      },
      t0->u)};
}

Allocator make(const parser::OmpClause::Allocator &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return Allocator{/*Allocator=*/makeExpr(inp.v, semaCtx)};
}

// AppendArgs: incomplete

At make(const parser::OmpClause::At &inp,
        semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: at");
}

// Never called, but needed for using "make" as a Clause visitor.
// See comment about "requires" clauses in Clauses.h.
AtomicDefaultMemOrder make(const parser::OmpClause::AtomicDefaultMemOrder &inp,
                           semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpAtomicDefaultMemOrderClause
  CLAUSET_ENUM_CONVERT( //
      convert, common::OmpAtomicDefaultMemOrderType,
      AtomicDefaultMemOrder::MemoryOrder,
      // clang-format off
      MS(AcqRel,   AcqRel)
      MS(Relaxed,  Relaxed)
      MS(SeqCst,   SeqCst)
      // clang-format on
  );

  return AtomicDefaultMemOrder{/*MemoryOrder=*/convert(inp.v.v)};
}

Bind make(const parser::OmpClause::Bind &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: bind");
}

// CancellationConstructType: empty
// Capture: empty

Collapse make(const parser::OmpClause::Collapse &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntConstantExpr
  return Collapse{/*N=*/makeExpr(inp.v, semaCtx)};
}

// Compare: empty
// Contains: missing-in-parser

Copyin make(const parser::OmpClause::Copyin &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Copyin{/*List=*/makeObjects(inp.v, semaCtx)};
}

Copyprivate make(const parser::OmpClause::Copyprivate &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Copyprivate{/*List=*/makeObjects(inp.v, semaCtx)};
}

Default make(const parser::OmpClause::Default &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDefaultClause
  using wrapped = parser::OmpDefaultClause;

  CLAUSET_ENUM_CONVERT( //
      convert, wrapped::Type, Default::DataSharingAttribute,
      // clang-format off
      MS(Firstprivate, Firstprivate)
      MS(None,         None)
      MS(Private,      Private)
      MS(Shared,       Shared)
      // clang-format on
  );

  return Default{/*DataSharingAttribute=*/convert(inp.v.v)};
}

Defaultmap make(const parser::OmpClause::Defaultmap &inp,
                semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDefaultmapClause
  using wrapped = parser::OmpDefaultmapClause;

  CLAUSET_ENUM_CONVERT( //
      convert1, wrapped::ImplicitBehavior, Defaultmap::ImplicitBehavior,
      // clang-format off
      MS(Alloc,        Alloc)
      MS(To,           To)
      MS(From,         From)
      MS(Tofrom,       Tofrom)
      MS(Firstprivate, Firstprivate)
      MS(None,         None)
      MS(Default,      Default)
      // MS(, Present)  missing-in-parser
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convert2, wrapped::VariableCategory, Defaultmap::VariableCategory,
      // clang-format off
      MS(Scalar,       Scalar)
      MS(Aggregate,    Aggregate)
      MS(Pointer,      Pointer)
      MS(Allocatable,  Allocatable)
      // clang-format on
  );

  auto &t0 = std::get<wrapped::ImplicitBehavior>(inp.v.t);
  auto &t1 = std::get<std::optional<wrapped::VariableCategory>>(inp.v.t);
  return Defaultmap{{/*ImplicitBehavior=*/convert1(t0),
                     /*VariableCategory=*/maybeApply(convert2, t1)}};
}

Depend make(const parser::OmpClause::Depend &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDependClause
  using wrapped = parser::OmpDependClause;
  using Variant = decltype(Depend::u);
  // Iteration is the equivalent of parser::OmpDependSinkVec
  using Iteration = Doacross::Vector::value_type; // LoopIterationT

  CLAUSET_ENUM_CONVERT( //
      convert1, parser::OmpDependenceType::Type, Depend::TaskDependenceType,
      // clang-format off
      MS(In,     In)
      MS(Out,    Out)
      MS(Inout,  Inout)
      // MS(, Mutexinoutset)   // missing-in-parser
      // MS(, Inputset)        // missing-in-parser
      // MS(, Depobj)          // missing-in-parser
      // clang-format on
  );

  return Depend{Fortran::common::visit( //
      common::visitors{
          // Doacross
          [&](const wrapped::Source &s) -> Variant {
            return Doacross{
                {/*DependenceType=*/Doacross::DependenceType::Source,
                 /*Vector=*/{}}};
          },
          // Doacross
          [&](const wrapped::Sink &s) -> Variant {
            using DependLength = parser::OmpDependSinkVecLength;
            auto convert2 = [&](const parser::OmpDependSinkVec &v) {
              auto &t0 = std::get<parser::Name>(v.t);
              auto &t1 = std::get<std::optional<DependLength>>(v.t);

              auto convert3 = [&](const DependLength &u) {
                auto &s0 = std::get<parser::DefinedOperator>(u.t);
                auto &s1 = std::get<parser::ScalarIntConstantExpr>(u.t);
                return Iteration::Distance{
                    {makeDefinedOperator(s0, semaCtx), makeExpr(s1, semaCtx)}};
              };
              return Iteration{
                  {makeObject(t0, semaCtx), maybeApply(convert3, t1)}};
            };
            return Doacross{{/*DependenceType=*/Doacross::DependenceType::Sink,
                             /*Vector=*/makeList(s.v, convert2)}};
          },
          // Depend::WithLocators
          [&](const wrapped::InOut &s) -> Variant {
            auto &t0 = std::get<parser::OmpDependenceType>(s.t);
            auto &t1 = std::get<std::list<parser::Designator>>(s.t);
            auto convert4 = [&](const parser::Designator &t) {
              return makeObject(t, semaCtx);
            };
            return Depend::WithLocators{
                {/*TaskDependenceType=*/convert1(t0.v),
                 /*Iterator=*/std::nullopt,
                 /*LocatorList=*/makeList(t1, convert4)}};
          },
      },
      inp.v.u)};
}

// Depobj: empty

Destroy make(const parser::OmpClause::Destroy &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: destroy");
}

Detach make(const parser::OmpClause::Detach &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: detach");
}

Device make(const parser::OmpClause::Device &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDeviceClause
  using wrapped = parser::OmpDeviceClause;

  CLAUSET_ENUM_CONVERT( //
      convert, parser::OmpDeviceClause::DeviceModifier, Device::DeviceModifier,
      // clang-format off
      MS(Ancestor,   Ancestor)
      MS(Device_Num, DeviceNum)
      // clang-format on
  );
  auto &t0 = std::get<std::optional<wrapped::DeviceModifier>>(inp.v.t);
  auto &t1 = std::get<parser::ScalarIntExpr>(inp.v.t);
  return Device{{/*DeviceModifier=*/maybeApply(convert, t0),
                 /*DeviceDescription=*/makeExpr(t1, semaCtx)}};
}

DeviceType make(const parser::OmpClause::DeviceType &inp,
                semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDeviceTypeClause
  using wrapped = parser::OmpDeviceTypeClause;

  CLAUSET_ENUM_CONVERT( //
      convert, wrapped::Type, DeviceType::DeviceTypeDescription,
      // clang-format off
      MS(Any,    Any)
      MS(Host,   Host)
      MS(Nohost, Nohost)
      // clang-format om
  );
  return DeviceType{/*DeviceTypeDescription=*/convert(inp.v.v)};
}

DistSchedule make(const parser::OmpClause::DistSchedule &inp,
                  semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::ScalarIntExpr>
  return DistSchedule{{/*Kind=*/DistSchedule::Kind::Static,
                       /*ChunkSize=*/maybeApply(makeExprFn(semaCtx), inp.v)}};
}

Doacross make(const parser::OmpClause::Doacross &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: doacross");
}

// DynamicAllocators: empty

Enter make(const parser::OmpClause::Enter &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Enter{makeObjects(/*List=*/inp.v, semaCtx)};
}

Exclusive make(const parser::OmpClause::Exclusive &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: exclusive");
}

Fail make(const parser::OmpClause::Fail &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: fail");
}

Filter make(const parser::OmpClause::Filter &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return Filter{/*ThreadNum=*/makeExpr(inp.v, semaCtx)};
}

Final make(const parser::OmpClause::Final &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarLogicalExpr
  return Final{/*Finalize=*/makeExpr(inp.v, semaCtx)};
}

Firstprivate make(const parser::OmpClause::Firstprivate &inp,
                  semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Firstprivate{/*List=*/makeObjects(inp.v, semaCtx)};
}

// Flush: empty

From make(const parser::OmpClause::From &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return From{{/*Expectation=*/std::nullopt, /*Mapper=*/std::nullopt,
               /*Iterator=*/std::nullopt,
               /*LocatorList=*/makeObjects(inp.v, semaCtx)}};
}

// Full: empty

Grainsize make(const parser::OmpClause::Grainsize &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return Grainsize{{/*Prescriptiveness=*/std::nullopt,
                    /*GrainSize=*/makeExpr(inp.v, semaCtx)}};
}

HasDeviceAddr make(const parser::OmpClause::HasDeviceAddr &inp,
                   semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return HasDeviceAddr{/*List=*/makeObjects(inp.v, semaCtx)};
}

Hint make(const parser::OmpClause::Hint &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ConstantExpr
  return Hint{/*HintExpr=*/makeExpr(inp.v, semaCtx)};
}

// Holds: missing-in-parser

If make(const parser::OmpClause::If &inp,
        semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpIfClause
  using wrapped = parser::OmpIfClause;

  CLAUSET_ENUM_CONVERT( //
      convert, wrapped::DirectiveNameModifier, llvm::omp::Directive,
      // clang-format off
      MS(Parallel,         OMPD_parallel)
      MS(Simd,             OMPD_simd)
      MS(Target,           OMPD_target)
      MS(TargetData,       OMPD_target_data)
      MS(TargetEnterData,  OMPD_target_enter_data)
      MS(TargetExitData,   OMPD_target_exit_data)
      MS(TargetUpdate,     OMPD_target_update)
      MS(Task,             OMPD_task)
      MS(Taskloop,         OMPD_taskloop)
      MS(Teams,            OMPD_teams)
      // clang-format on
  );
  auto &t0 = std::get<std::optional<wrapped::DirectiveNameModifier>>(inp.v.t);
  auto &t1 = std::get<parser::ScalarLogicalExpr>(inp.v.t);
  return If{{/*DirectiveNameModifier=*/maybeApply(convert, t0),
             /*IfExpression=*/makeExpr(t1, semaCtx)}};
}

// Inbranch: empty

Inclusive make(const parser::OmpClause::Inclusive &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: inclusive");
}

Indirect make(const parser::OmpClause::Indirect &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: indirect");
}

Init make(const parser::OmpClause::Init &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: init");
}

// Initializer: missing-in-parser

InReduction make(const parser::OmpClause::InReduction &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpInReductionClause
  auto &t0 = std::get<parser::OmpReductionOperator>(inp.v.t);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);
  return InReduction{
      {/*ReductionIdentifiers=*/{makeReductionOperator(t0, semaCtx)},
       /*List=*/makeObjects(t1, semaCtx)}};
}

IsDevicePtr make(const parser::OmpClause::IsDevicePtr &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return IsDevicePtr{/*List=*/makeObjects(inp.v, semaCtx)};
}

Lastprivate make(const parser::OmpClause::Lastprivate &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Lastprivate{{/*LastprivateModifier=*/std::nullopt,
                      /*List=*/makeObjects(inp.v, semaCtx)}};
}

Linear make(const parser::OmpClause::Linear &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpLinearClause
  using wrapped = parser::OmpLinearClause;

  CLAUSET_ENUM_CONVERT( //
      convert, parser::OmpLinearModifier::Type, Linear::LinearModifier,
      // clang-format off
      MS(Ref,  Ref)
      MS(Val,  Val)
      MS(Uval, Uval)
      // clang-format on
  );

  using Tuple = decltype(Linear::t);

  return Linear{Fortran::common::visit(
      common::visitors{
          [&](const wrapped::WithModifier &s) -> Tuple {
            return {
                /*StepSimpleModifier=*/std::nullopt,
                /*StepComplexModifier=*/maybeApply(makeExprFn(semaCtx), s.step),
                /*LinearModifier=*/convert(s.modifier.v),
                /*List=*/makeList(s.names, makeObjectFn(semaCtx))};
          },
          [&](const wrapped::WithoutModifier &s) -> Tuple {
            return {
                /*StepSimpleModifier=*/maybeApply(makeExprFn(semaCtx), s.step),
                /*StepComplexModifier=*/std::nullopt,
                /*LinearModifier=*/std::nullopt,
                /*List=*/makeList(s.names, makeObjectFn(semaCtx))};
          },
      },
      inp.v.u)};
}

Link make(const parser::OmpClause::Link &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Link{/*List=*/makeObjects(inp.v, semaCtx)};
}

Map make(const parser::OmpClause::Map &inp,
         semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpMapClause

  CLAUSET_ENUM_CONVERT( //
      convert1, parser::OmpMapType::Type, Map::MapType,
      // clang-format off
      MS(To,       To)
      MS(From,     From)
      MS(Tofrom,   Tofrom)
      MS(Alloc,    Alloc)
      MS(Release,  Release)
      MS(Delete,   Delete)
      // clang-format on
  );

  // No convert2: MapTypeModifier is not an enum in parser.

  auto &t0 = std::get<std::optional<parser::OmpMapType>>(inp.v.t);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);

  if (!t0) {
    return Map{{/*MapType=*/std::nullopt, /*MapTypeModifiers=*/std::nullopt,
                /*Mapper=*/std::nullopt, /*Iterator=*/std::nullopt,
                /*LocatorList=*/makeObjects(t1, semaCtx)}};
  }

  auto &s0 = std::get<std::optional<parser::OmpMapType::Always>>(t0->t);
  auto &s1 = std::get<parser::OmpMapType::Type>(t0->t);

  std::optional<Map::MapTypeModifiers> maybeList;
  if (s0)
    maybeList = Map::MapTypeModifiers{Map::MapTypeModifier::Always};

  return Map{{/*MapType=*/convert1(s1),
              /*MapTypeModifiers=*/maybeList,
              /*Mapper=*/std::nullopt, /*Iterator=*/std::nullopt,
              /*LocatorList=*/makeObjects(t1, semaCtx)}};
}

// Match: incomplete
// MemoryOrder: empty
// Mergeable: empty

Message make(const parser::OmpClause::Message &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: message");
}

Nocontext make(const parser::OmpClause::Nocontext &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarLogicalExpr
  return Nocontext{/*DoNotUpdateContext=*/makeExpr(inp.v, semaCtx)};
}

// Nogroup: empty

Nontemporal make(const parser::OmpClause::Nontemporal &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::list<parser::Name>
  return Nontemporal{/*List=*/makeList(inp.v, makeObjectFn(semaCtx))};
}

// NoOpenmp: missing-in-parser
// NoOpenmpRoutines: missing-in-parser
// NoParallelism: missing-in-parser
// Notinbranch: empty

Novariants make(const parser::OmpClause::Novariants &inp,
                semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarLogicalExpr
  return Novariants{/*DoNotUseVariant=*/makeExpr(inp.v, semaCtx)};
}

// Nowait: empty

NumTasks make(const parser::OmpClause::NumTasks &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return NumTasks{{/*Prescriptiveness=*/std::nullopt,
                   /*NumTasks=*/makeExpr(inp.v, semaCtx)}};
}

NumTeams make(const parser::OmpClause::NumTeams &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return NumTeams{{/*LowerBound=*/std::nullopt,
                   /*UpperBound=*/makeExpr(inp.v, semaCtx)}};
}

NumThreads make(const parser::OmpClause::NumThreads &inp,
                semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return NumThreads{/*Nthreads=*/makeExpr(inp.v, semaCtx)};
}

// OmpxAttribute: empty
// OmpxBare: empty

OmpxDynCgroupMem make(const parser::OmpClause::OmpxDynCgroupMem &inp,
                      semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return OmpxDynCgroupMem{makeExpr(inp.v, semaCtx)};
}

Order make(const parser::OmpClause::Order &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpOrderClause
  using wrapped = parser::OmpOrderClause;

  CLAUSET_ENUM_CONVERT( //
      convert1, parser::OmpOrderModifier::Kind, Order::OrderModifier,
      // clang-format off
      MS(Reproducible,   Reproducible)
      MS(Unconstrained,  Unconstrained)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convert2, wrapped::Type, Order::Ordering,
      // clang-format off
      MS(Concurrent, Concurrent)
      // clang-format on
  );

  auto &t0 = std::get<std::optional<parser::OmpOrderModifier>>(inp.v.t);
  auto &t1 = std::get<wrapped::Type>(inp.v.t);

  auto convert3 = [&](const parser::OmpOrderModifier &s) {
    return Fortran::common::visit(
        [&](parser::OmpOrderModifier::Kind k) { return convert1(k); }, s.u);
  };
  return Order{
      {/*OrderModifier=*/maybeApply(convert3, t0), /*Ordering=*/convert2(t1)}};
}

Ordered make(const parser::OmpClause::Ordered &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::ScalarIntConstantExpr>
  return Ordered{/*N=*/maybeApply(makeExprFn(semaCtx), inp.v)};
}

// Otherwise: incomplete, missing-in-parser

Partial make(const parser::OmpClause::Partial &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::ScalarIntConstantExpr>
  return Partial{/*UnrollFactor=*/maybeApply(makeExprFn(semaCtx), inp.v)};
}

Priority make(const parser::OmpClause::Priority &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return Priority{/*PriorityValue=*/makeExpr(inp.v, semaCtx)};
}

Private make(const parser::OmpClause::Private &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Private{/*List=*/makeObjects(inp.v, semaCtx)};
}

ProcBind make(const parser::OmpClause::ProcBind &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpProcBindClause
  using wrapped = parser::OmpProcBindClause;

  CLAUSET_ENUM_CONVERT( //
      convert, wrapped::Type, ProcBind::AffinityPolicy,
      // clang-format off
      MS(Close,    Close)
      MS(Master,   Master)
      MS(Spread,   Spread)
      MS(Primary,  Primary)
      // clang-format on
  );
  return ProcBind{/*AffinityPolicy=*/convert(inp.v.v)};
}

// Read: empty

Reduction make(const parser::OmpClause::Reduction &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpReductionClause
  using wrapped = parser::OmpReductionClause;

  CLAUSET_ENUM_CONVERT( //
      convert, wrapped::ReductionModifier, Reduction::ReductionModifier,
      // clang-format off
      MS(Inscan,  Inscan)
      MS(Task,    Task)
      MS(Default, Default)
      // clang-format on
  );

  auto &t0 =
      std::get<std::optional<parser::OmpReductionClause::ReductionModifier>>(
          inp.v.t);
  auto &t1 = std::get<parser::OmpReductionOperator>(inp.v.t);
  auto &t2 = std::get<parser::OmpObjectList>(inp.v.t);
  return Reduction{
      {/*ReductionModifier=*/t0
           ? std::make_optional<Reduction::ReductionModifier>(convert(*t0))
           : std::nullopt,
       /*ReductionIdentifiers=*/{makeReductionOperator(t1, semaCtx)},
       /*List=*/makeObjects(t2, semaCtx)}};
}

// Relaxed: empty
// Release: empty
// ReverseOffload: empty

Safelen make(const parser::OmpClause::Safelen &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntConstantExpr
  return Safelen{/*Length=*/makeExpr(inp.v, semaCtx)};
}

Schedule make(const parser::OmpClause::Schedule &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpScheduleClause
  using wrapped = parser::OmpScheduleClause;

  CLAUSET_ENUM_CONVERT( //
      convert1, wrapped::ScheduleType, Schedule::Kind,
      // clang-format off
      MS(Static,   Static)
      MS(Dynamic,  Dynamic)
      MS(Guided,   Guided)
      MS(Auto,     Auto)
      MS(Runtime,  Runtime)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convert2, parser::OmpScheduleModifierType::ModType,
      Schedule::OrderingModifier,
      // clang-format off
      MS(Monotonic,    Monotonic)
      MS(Nonmonotonic, Nonmonotonic)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convert3, parser::OmpScheduleModifierType::ModType,
      Schedule::ChunkModifier,
      // clang-format off
      MS(Simd, Simd)
      // clang-format on
  );

  auto &t0 = std::get<std::optional<parser::OmpScheduleModifier>>(inp.v.t);
  auto &t1 = std::get<wrapped::ScheduleType>(inp.v.t);
  auto &t2 = std::get<std::optional<parser::ScalarIntExpr>>(inp.v.t);

  if (!t0) {
    return Schedule{{/*Kind=*/convert1(t1), /*OrderingModifier=*/std::nullopt,
                     /*ChunkModifier=*/std::nullopt,
                     /*ChunkSize=*/maybeApply(makeExprFn(semaCtx), t2)}};
  }

  // The members of parser::OmpScheduleModifier correspond to OrderingModifier,
  // and ChunkModifier, but they can appear in any order.
  auto &m1 = std::get<parser::OmpScheduleModifier::Modifier1>(t0->t);
  auto &m2 =
      std::get<std::optional<parser::OmpScheduleModifier::Modifier2>>(t0->t);

  std::optional<Schedule::OrderingModifier> omod;
  std::optional<Schedule::ChunkModifier> cmod;

  if (m1.v.v == parser::OmpScheduleModifierType::ModType::Simd) {
    // m1 is chunk-modifier
    cmod = convert3(m1.v.v);
    if (m2)
      omod = convert2(m2->v.v);
  } else {
    // m1 is ordering-modifier
    omod = convert2(m1.v.v);
    if (m2)
      cmod = convert3(m2->v.v);
  }

  return Schedule{{/*Kind=*/convert1(t1),
                   /*OrderingModifier=*/omod,
                   /*ChunkModifier=*/cmod,
                   /*ChunkSize=*/maybeApply(makeExprFn(semaCtx), t2)}};
}

// SeqCst: empty

Severity make(const parser::OmpClause::Severity &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: severity");
}

Shared make(const parser::OmpClause::Shared &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Shared{/*List=*/makeObjects(inp.v, semaCtx)};
}

// Simd: empty

Simdlen make(const parser::OmpClause::Simdlen &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntConstantExpr
  return Simdlen{/*Length=*/makeExpr(inp.v, semaCtx)};
}

Sizes make(const parser::OmpClause::Sizes &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::list<parser::ScalarIntExpr>
  return Sizes{/*SizeList=*/makeList(inp.v, makeExprFn(semaCtx))};
}

TaskReduction make(const parser::OmpClause::TaskReduction &inp,
                   semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpReductionClause
  auto &t0 = std::get<parser::OmpReductionOperator>(inp.v.t);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);
  return TaskReduction{
      {/*ReductionIdentifiers=*/{makeReductionOperator(t0, semaCtx)},
       /*List=*/makeObjects(t1, semaCtx)}};
}

ThreadLimit make(const parser::OmpClause::ThreadLimit &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return ThreadLimit{/*Threadlim=*/makeExpr(inp.v, semaCtx)};
}

// Threadprivate: empty
// Threads: empty

To make(const parser::OmpClause::To &inp,
        semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return To{{/*Expectation=*/std::nullopt, /*Mapper=*/std::nullopt,
             /*Iterator=*/std::nullopt,
             /*LocatorList=*/makeObjects(inp.v, semaCtx)}};
}

// UnifiedAddress: empty
// UnifiedSharedMemory: empty

Uniform make(const parser::OmpClause::Uniform &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::list<parser::Name>
  return Uniform{/*ParameterList=*/makeList(inp.v, makeObjectFn(semaCtx))};
}

// Unknown: empty
// Untied: empty

Update make(const parser::OmpClause::Update &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  return Update{/*TaskDependenceType=*/std::nullopt};
}

Use make(const parser::OmpClause::Use &inp,
         semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: use");
}

UseDeviceAddr make(const parser::OmpClause::UseDeviceAddr &inp,
                   semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return UseDeviceAddr{/*List=*/makeObjects(inp.v, semaCtx)};
}

UseDevicePtr make(const parser::OmpClause::UseDevicePtr &inp,
                  semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return UseDevicePtr{/*List=*/makeObjects(inp.v, semaCtx)};
}

UsesAllocators make(const parser::OmpClause::UsesAllocators &inp,
                    semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: uses_allocators");
}

// Weak: empty
// When: incomplete
// Write: empty
} // namespace clause

Clause makeClause(const parser::OmpClause &cls,
                  semantics::SemanticsContext &semaCtx) {
  return Fortran::common::visit(
      [&](auto &&s) {
        return makeClause(getClauseId(cls), clause::make(s, semaCtx),
                          cls.source);
      },
      cls.u);
}

List<Clause> makeClauses(const parser::OmpClauseList &clauses,
                         semantics::SemanticsContext &semaCtx) {
  return makeList(clauses.v, [&](const parser::OmpClause &s) {
    return makeClause(s, semaCtx);
  });
}

bool transferLocations(const List<Clause> &from, List<Clause> &to) {
  bool allDone = true;

  for (Clause &clause : to) {
    if (!clause.source.empty())
      continue;
    auto found =
        llvm::find_if(from, [&](const Clause &c) { return c.id == clause.id; });
    // This is not completely accurate, but should be good enough for now.
    // It can be improved in the future if necessary, but in cases of
    // synthesized clauses getting accurate location may be impossible.
    if (found != from.end()) {
      clause.source = found->source;
    } else {
      // Found a clause that won't have "source".
      allDone = false;
    }
  }

  return allDone;
}

} // namespace Fortran::lower::omp
