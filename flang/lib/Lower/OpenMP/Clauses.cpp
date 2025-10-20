//===-- Clauses.cpp -- OpenMP clause handling -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OpenMP/Clauses.h"

#include "flang/Common/idioms.h"
#include "flang/Evaluate/expression.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/openmp-modifiers.h"
#include "flang/Semantics/symbol.h"

#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include <list>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>

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
    const std::optional<evaluate::Expr<evaluate::SomeType>> &maybeDsg =
        std::get<1>(sd);
    if (!maybeDsg)
      return; // Symbol with no designator -> OK
    assert(symbol && "Expecting symbol");
    std::optional<evaluate::DataRef> maybeRef = evaluate::ExtractDataRef(
        *maybeDsg, /*intoSubstring=*/true, /*intoComplexPart=*/true);
    if (maybeRef) {
      if (&maybeRef->GetLastSymbol() == symbol)
        return; // Symbol with a designator for it -> OK
      llvm_unreachable("Expecting designator for given symbol");
    } else {
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

ObjectList makeObjects(const parser::OmpArgumentList &objects,
                       semantics::SemanticsContext &semaCtx) {
  return makeList(objects.v, [&](const parser::OmpArgument &arg) {
    return common::visit(
        common::visitors{
            [&](const parser::OmpLocator &locator) -> Object {
              if (auto *object = std::get_if<parser::OmpObject>(&locator.u)) {
                return makeObject(*object, semaCtx);
              }
              llvm_unreachable("Expecting object");
            },
            [](auto &&s) -> Object { //
              llvm_unreachable("Expecting object");
            },
        },
        arg.u);
  });
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
    } else if (auto *symRef = base.UnwrapSymbolRef()) {
      // This is the base symbol of the array reference, which is the same
      // as the symbol in the input object,
      // e.g. A(i) is represented as {Symbol(A), Designator(ArrayRef(A, i))}.
      // Here we have the Symbol(A), which is what we started with.
      (void)symRef;
      assert(&**symRef == object.sym());
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
MAKE_EMPTY_CLASS(Full, Full);
MAKE_EMPTY_CLASS(Inbranch, Inbranch);
MAKE_EMPTY_CLASS(Mergeable, Mergeable);
MAKE_EMPTY_CLASS(Nogroup, Nogroup);
MAKE_EMPTY_CLASS(NoOpenmp, NoOpenmp);
MAKE_EMPTY_CLASS(NoOpenmpRoutines, NoOpenmpRoutines);
MAKE_EMPTY_CLASS(NoOpenmpConstructs, NoOpenmpConstructs);
MAKE_EMPTY_CLASS(NoParallelism, NoParallelism);
MAKE_EMPTY_CLASS(Notinbranch, Notinbranch);
MAKE_EMPTY_CLASS(Nowait, Nowait);
MAKE_EMPTY_CLASS(OmpxAttribute, OmpxAttribute);
MAKE_EMPTY_CLASS(OmpxBare, OmpxBare);
MAKE_EMPTY_CLASS(Read, Read);
MAKE_EMPTY_CLASS(Relaxed, Relaxed);
MAKE_EMPTY_CLASS(Release, Release);
MAKE_EMPTY_CLASS(SeqCst, SeqCst);
MAKE_EMPTY_CLASS(Simd, Simd);
MAKE_EMPTY_CLASS(Threads, Threads);
MAKE_EMPTY_CLASS(Unknown, Unknown);
MAKE_EMPTY_CLASS(Untied, Untied);
MAKE_EMPTY_CLASS(Weak, Weak);
MAKE_EMPTY_CLASS(Write, Write);

// Artificial clauses
MAKE_EMPTY_CLASS(Depobj, Depobj);
MAKE_EMPTY_CLASS(Flush, Flush);
MAKE_EMPTY_CLASS(MemoryOrder, MemoryOrder);
MAKE_EMPTY_CLASS(Threadprivate, Threadprivate);
MAKE_EMPTY_CLASS(Groupprivate, Groupprivate);

MAKE_INCOMPLETE_CLASS(AdjustArgs, AdjustArgs);
MAKE_INCOMPLETE_CLASS(AppendArgs, AppendArgs);
MAKE_INCOMPLETE_CLASS(GraphId, GraphId);
MAKE_INCOMPLETE_CLASS(GraphReset, GraphReset);
MAKE_INCOMPLETE_CLASS(Replayable, Replayable);
MAKE_INCOMPLETE_CLASS(Transparent, Transparent);

List<IteratorSpecifier>
makeIteratorSpecifiers(const parser::OmpIteratorSpecifier &inp,
                       semantics::SemanticsContext &semaCtx) {
  List<IteratorSpecifier> specifiers;

  auto &[begin, end, step] = std::get<parser::SubscriptTriplet>(inp.t).t;
  assert(begin && end && "Expecting begin/end values");
  evaluate::ExpressionAnalyzer ea{semaCtx};

  MaybeExpr rbegin{ea.Analyze(*begin)}, rend{ea.Analyze(*end)};
  MaybeExpr rstep;
  if (step)
    rstep = ea.Analyze(*step);

  assert(rbegin && rend && "Unable to get range bounds");
  Range range{{*rbegin, *rend, rstep}};

  auto &tds = std::get<parser::TypeDeclarationStmt>(inp.t);
  auto &entities = std::get<std::list<parser::EntityDecl>>(tds.t);
  for (const parser::EntityDecl &ed : entities) {
    auto &name = std::get<parser::ObjectName>(ed.t);
    assert(name.symbol && "Expecting symbol for iterator variable");
    auto *stype = name.symbol->GetType();
    assert(stype && "Expecting symbol type");
    IteratorSpecifier spec{{evaluate::DynamicType::From(*stype),
                            makeObject(name, semaCtx), range}};
    specifiers.emplace_back(std::move(spec));
  }

  return specifiers;
}

Iterator makeIterator(const parser::OmpIterator &inp,
                      semantics::SemanticsContext &semaCtx) {
  Iterator iterator;
  for (auto &&spec : inp.v)
    llvm::append_range(iterator, makeIteratorSpecifiers(spec, semaCtx));
  return iterator;
}

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

ReductionOperator
makeReductionOperator(const parser::OmpReductionIdentifier &inp,
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

clause::DependenceType makeDepType(const parser::OmpDependenceType &inp) {
  switch (inp.v) {
  case parser::OmpDependenceType::Value::Sink:
    return clause::DependenceType::Sink;
  case parser::OmpDependenceType::Value::Source:
    return clause::DependenceType::Source;
  }
  llvm_unreachable("Unexpected dependence type");
}

clause::DependenceType makeDepType(const parser::OmpTaskDependenceType &inp) {
  switch (inp.v) {
  case parser::OmpTaskDependenceType::Value::Depobj:
    return clause::DependenceType::Depobj;
  case parser::OmpTaskDependenceType::Value::In:
    return clause::DependenceType::In;
  case parser::OmpTaskDependenceType::Value::Inout:
    return clause::DependenceType::Inout;
  case parser::OmpTaskDependenceType::Value::Inoutset:
    return clause::DependenceType::Inoutset;
  case parser::OmpTaskDependenceType::Value::Mutexinoutset:
    return clause::DependenceType::Mutexinoutset;
  case parser::OmpTaskDependenceType::Value::Out:
    return clause::DependenceType::Out;
  }
  llvm_unreachable("Unexpected task dependence type");
}

clause::Prescriptiveness
makePrescriptiveness(parser::OmpPrescriptiveness::Value v) {
  switch (v) {
  case parser::OmpPrescriptiveness::Value::Strict:
    return clause::Prescriptiveness::Strict;
  case parser::OmpPrescriptiveness::Value::Fallback:
    return clause::Prescriptiveness::Fallback;
  }
  llvm_unreachable("Unexpected prescriptiveness");
}

// --------------------------------------------------------------------
// Actual clauses. Each T (where tomp::T exists in ClauseT) has its "make".

Absent make(const parser::OmpClause::Absent &inp,
            semantics::SemanticsContext &semaCtx) {
  llvm_unreachable("Unimplemented: absent");
}

// AcqRel: empty
// Acquire: empty
// AdjustArgs: incomplete

Affinity make(const parser::OmpClause::Affinity &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpAffinityClause
  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 = semantics::OmpGetUniqueModifier<parser::OmpIterator>(mods);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);

  auto &&maybeIter =
      m0 ? makeIterator(*m0, semaCtx) : std::optional<Iterator>{};

  return Affinity{{/*Iterator=*/std::move(maybeIter),
                   /*LocatorList=*/makeObjects(t1, semaCtx)}};
}

Align make(const parser::OmpClause::Align &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: align");
}

Aligned make(const parser::OmpClause::Aligned &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpAlignedClause
  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto &t0 = std::get<parser::OmpObjectList>(inp.v.t);
  auto *m1 = semantics::OmpGetUniqueModifier<parser::OmpAlignment>(mods);

  return Aligned{{
      /*Alignment=*/maybeApplyToV(makeExprFn(semaCtx), m1),
      /*List=*/makeObjects(t0, semaCtx),
  }};
}

Allocate make(const parser::OmpClause::Allocate &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpAllocateClause
  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 = semantics::OmpGetUniqueModifier<parser::OmpAlignModifier>(mods);
  auto *m1 =
      semantics::OmpGetUniqueModifier<parser::OmpAllocatorComplexModifier>(
          mods);
  auto *m2 =
      semantics::OmpGetUniqueModifier<parser::OmpAllocatorSimpleModifier>(mods);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);

  auto makeAllocator = [&](auto *mod) -> std::optional<Allocator> {
    if (mod)
      return Allocator{makeExpr(mod->v, semaCtx)};
    return std::nullopt;
  };

  auto makeAlign = [&](const parser::ScalarIntExpr &expr) {
    return Align{makeExpr(expr, semaCtx)};
  };

  auto maybeAllocator = m1 ? makeAllocator(m1) : makeAllocator(m2);
  return Allocate{{/*AllocatorComplexModifier=*/std::move(maybeAllocator),
                   /*AlignModifier=*/maybeApplyToV(makeAlign, m0),
                   /*List=*/makeObjects(t1, semaCtx)}};
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
      convert, common::OmpMemoryOrderType, AtomicDefaultMemOrder::MemoryOrder,
      // clang-format off
      MS(Acq_Rel,  AcqRel)
      MS(Acquire,  Acquire)
      MS(Relaxed,  Relaxed)
      MS(Release,  Release)
      MS(Seq_Cst,  SeqCst)
      // clang-format on
  );

  return AtomicDefaultMemOrder{/*MemoryOrder=*/convert(inp.v.v)};
}

Bind make(const parser::OmpClause::Bind &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpBindClause
  using wrapped = parser::OmpBindClause;

  CLAUSET_ENUM_CONVERT( //
      convert, wrapped::Binding, Bind::Binding,
      // clang-format off
      MS(Teams, Teams)
      MS(Parallel, Parallel)
      MS(Thread, Thread)
      // clang-format on
  );

  return Bind{/*Binding=*/convert(inp.v.v)};
}

CancellationConstructType
make(const parser::OmpClause::CancellationConstructType &inp,
     semantics::SemanticsContext &semaCtx) {
  auto name = std::get<parser::OmpDirectiveName>(inp.v.t);
  CLAUSET_ENUM_CONVERT(
      convert, llvm::omp::Directive, llvm::omp::CancellationConstructType,
      // clang-format off
      MS(OMPD_parallel, OMP_CANCELLATION_CONSTRUCT_Parallel)
      MS(OMPD_do, OMP_CANCELLATION_CONSTRUCT_Loop)
      MS(OMPD_sections, OMP_CANCELLATION_CONSTRUCT_Sections)
      MS(OMPD_taskgroup, OMP_CANCELLATION_CONSTRUCT_Taskgroup)
      // clang-format on
  );

  return CancellationConstructType{convert(name.v)};
}

// Capture: empty

Collapse make(const parser::OmpClause::Collapse &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntConstantExpr
  return Collapse{/*N=*/makeExpr(inp.v, semaCtx)};
}

// Compare: empty

Contains make(const parser::OmpClause::Contains &inp,
              semantics::SemanticsContext &semaCtx) {
  llvm_unreachable("Unimplemented: contains");
}

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

// The Default clause is overloaded in OpenMP 5.0 and 5.1: it can be either
// a data-sharing clause, or a METADIRECTIVE clause. In the latter case, it
// has been superseded by the OTHERWISE clause.
// Disambiguate this in this representation: for the DSA case, create Default,
// and in the other case create Otherwise.
Default makeDefault(const parser::OmpClause::Default &inp,
                    semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDefaultClause
  using wrapped = parser::OmpDefaultClause;

  CLAUSET_ENUM_CONVERT( //
      convert, wrapped::DataSharingAttribute, Default::DataSharingAttribute,
      // clang-format off
      MS(Firstprivate, Firstprivate)
      MS(None,         None)
      MS(Private,      Private)
      MS(Shared,       Shared)
      // clang-format on
  );

  auto dsa = std::get<wrapped::DataSharingAttribute>(inp.v.u);
  return Default{/*DataSharingAttribute=*/convert(dsa)};
}

Otherwise makeOtherwise(const parser::OmpClause::Default &inp,
                        semantics::SemanticsContext &semaCtx) {
  return Otherwise{};
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
      MS(Present,      Present)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convert2, parser::OmpVariableCategory::Value,
      Defaultmap::VariableCategory,
      // clang-format off
      MS(Aggregate,    Aggregate)
      MS(All,          All)
      MS(Allocatable,  Allocatable)
      MS(Pointer,      Pointer)
      MS(Scalar,       Scalar)
      // clang-format on
  );

  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto &t0 = std::get<wrapped::ImplicitBehavior>(inp.v.t);
  auto *t1 = semantics::OmpGetUniqueModifier<parser::OmpVariableCategory>(mods);

  auto category = t1 ? convert2(t1->v) : Defaultmap::VariableCategory::All;
  return Defaultmap{{/*ImplicitBehavior=*/convert1(t0),
                     /*VariableCategory=*/category}};
}

Doacross makeDoacross(const parser::OmpDoacross &doa,
                      semantics::SemanticsContext &semaCtx) {
  // Iteration is the equivalent of parser::OmpIteration
  using Iteration = Doacross::Vector::value_type; // LoopIterationT

  auto visitSource = [&](const parser::OmpDoacross::Source &) {
    return Doacross{{/*DependenceType=*/Doacross::DependenceType::Source,
                     /*Vector=*/{}}};
  };

  auto visitSink = [&](const parser::OmpDoacross::Sink &s) {
    using IterOffset = parser::OmpIterationOffset;
    auto convert2 = [&](const parser::OmpIteration &v) {
      auto &t0 = std::get<parser::Name>(v.t);
      auto &t1 = std::get<std::optional<IterOffset>>(v.t);

      auto convert3 = [&](const IterOffset &u) {
        auto &s0 = std::get<parser::DefinedOperator>(u.t);
        auto &s1 = std::get<parser::ScalarIntConstantExpr>(u.t);
        return Iteration::Distance{
            {makeDefinedOperator(s0, semaCtx), makeExpr(s1, semaCtx)}};
      };
      return Iteration{{makeObject(t0, semaCtx), maybeApply(convert3, t1)}};
    };
    return Doacross{{/*DependenceType=*/Doacross::DependenceType::Sink,
                     /*Vector=*/makeList(s.v.v, convert2)}};
  };

  return common::visit(common::visitors{visitSink, visitSource}, doa.u);
}

Depend make(const parser::OmpClause::Depend &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDependClause
  using wrapped = parser::OmpDependClause;
  using Variant = decltype(Depend::u);

  auto visitTaskDep = [&](const wrapped::TaskDep &s) -> Variant {
    auto &mods = semantics::OmpGetModifiers(s);
    auto *m0 = semantics::OmpGetUniqueModifier<parser::OmpIterator>(mods);
    auto *m1 =
        semantics::OmpGetUniqueModifier<parser::OmpTaskDependenceType>(mods);
    auto &t1 = std::get<parser::OmpObjectList>(s.t);
    assert(m1 && "expecting task dependence type");

    auto &&maybeIter =
        m0 ? makeIterator(*m0, semaCtx) : std::optional<Iterator>{};
    return Depend::TaskDep{{/*DependenceType=*/makeDepType(*m1),
                            /*Iterator=*/std::move(maybeIter),
                            /*LocatorList=*/makeObjects(t1, semaCtx)}};
  };

  return Depend{common::visit( //
      common::visitors{
          // Doacross
          [&](const parser::OmpDoacross &s) -> Variant {
            return makeDoacross(s, semaCtx);
          },
          // Depend::TaskDep
          visitTaskDep,
      },
      inp.v.u)};
}

// Depobj: empty

Destroy make(const parser::OmpClause::Destroy &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<OmpDestroyClause>
  auto &&maybeObject = maybeApply(
      [&](const parser::OmpDestroyClause &c) {
        return makeObject(c.v, semaCtx);
      },
      inp.v);

  return Destroy{/*DestroyVar=*/std::move(maybeObject)};
}

Detach make(const parser::OmpClause::Detach &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDetachClause
  return Detach{makeObject(inp.v.v, semaCtx)};
}

Device make(const parser::OmpClause::Device &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDeviceClause
  CLAUSET_ENUM_CONVERT( //
      convert, parser::OmpDeviceModifier::Value, Device::DeviceModifier,
      // clang-format off
      MS(Ancestor,   Ancestor)
      MS(Device_Num, DeviceNum)
      // clang-format on
  );

  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 = semantics::OmpGetUniqueModifier<parser::OmpDeviceModifier>(mods);
  auto &t1 = std::get<parser::ScalarIntExpr>(inp.v.t);
  return Device{{/*DeviceModifier=*/maybeApplyToV(convert, m0),
                 /*DeviceDescription=*/makeExpr(t1, semaCtx)}};
}

DeviceSafesync make(const parser::OmpClause::DeviceSafesync &inp,
                    semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::OmpDeviceSafesyncClause>
  auto &&maybeRequired = maybeApply(
      [&](const parser::OmpDeviceSafesyncClause &c) {
        return makeExpr(c.v, semaCtx);
      },
      inp.v);

  return DeviceSafesync{/*Required=*/std::move(maybeRequired)};
}

DeviceType make(const parser::OmpClause::DeviceType &inp,
                semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDeviceTypeClause
  using wrapped = parser::OmpDeviceTypeClause;

  CLAUSET_ENUM_CONVERT( //
      convert, wrapped::DeviceTypeDescription,
      DeviceType::DeviceTypeDescription,
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
  // inp.v -> OmpDoacrossClause
  return makeDoacross(inp.v.v, semaCtx);
}

DynamicAllocators make(const parser::OmpClause::DynamicAllocators &inp,
                       semantics::SemanticsContext &semaCtx) {
  // inp.v -> td::optional<arser::OmpDynamicAllocatorsClause>
  auto &&maybeRequired = maybeApply(
      [&](const parser::OmpDynamicAllocatorsClause &c) {
        return makeExpr(c.v, semaCtx);
      },
      inp.v);

  return DynamicAllocators{/*Required=*/std::move(maybeRequired)};
}


DynGroupprivate make(const parser::OmpClause::DynGroupprivate &inp,
                     semantics::SemanticsContext &semaCtx) {
  // imp.v -> OmpDyngroupprivateClause
  CLAUSET_ENUM_CONVERT( //
      convert, parser::OmpAccessGroup::Value, DynGroupprivate::AccessGroup,
      // clang-format off
      MS(Cgroup,  Cgroup)
      // clang-format on
  );

  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 = semantics::OmpGetUniqueModifier<parser::OmpAccessGroup>(mods);
  auto *m1 = semantics::OmpGetUniqueModifier<parser::OmpPrescriptiveness>(mods);
  auto &size = std::get<parser::ScalarIntExpr>(inp.v.t);

  return DynGroupprivate{
      {/*AccessGroup=*/maybeApplyToV(convert, m0),
       /*Prescriptiveness=*/maybeApplyToV(makePrescriptiveness, m1),
       /*Size=*/makeExpr(size, semaCtx)}};
}

Enter make(const parser::OmpClause::Enter &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpEnterClause
  CLAUSET_ENUM_CONVERT( //
      convert, parser::OmpAutomapModifier::Value, Enter::Modifier,
      // clang-format off
      MS(Automap, Automap)
      // clang-format on
  );
  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *mod = semantics::OmpGetUniqueModifier<parser::OmpAutomapModifier>(mods);
  auto &objList = std::get<parser::OmpObjectList>(inp.v.t);

  return Enter{{/*Modifier=*/maybeApplyToV(convert, mod),
                /*List=*/makeObjects(objList, semaCtx)}};
}

Exclusive make(const parser::OmpClause::Exclusive &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Exclusive{makeObjects(/*List=*/inp.v, semaCtx)};
}

Fail make(const parser::OmpClause::Fail &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpFalClause
  CLAUSET_ENUM_CONVERT( //
      convert, common::OmpMemoryOrderType, Fail::MemoryOrder,
      // clang-format off
      MS(Acq_Rel,  AcqRel)
      MS(Acquire,  Acquire)
      MS(Relaxed,  Relaxed)
      MS(Release,  Release)
      MS(Seq_Cst,  SeqCst)
      // clang-format on
  );

  return Fail{/*MemoryOrder=*/convert(inp.v.v)};
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
  // inp.v -> parser::OmpFromClause
  CLAUSET_ENUM_CONVERT( //
      convert, parser::OmpExpectation::Value, From::Expectation,
      // clang-format off
      MS(Present, Present)
      // clang-format on
  );

  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *t0 = semantics::OmpGetUniqueModifier<parser::OmpExpectation>(mods);
  auto *t1 = semantics::OmpGetUniqueModifier<parser::OmpMapper>(mods);
  auto *t2 = semantics::OmpGetUniqueModifier<parser::OmpIterator>(mods);
  auto &t3 = std::get<parser::OmpObjectList>(inp.v.t);

  auto mappers = [&]() -> std::optional<List<Mapper>> {
    if (t1)
      return List<Mapper>{Mapper{makeObject(t1->v, semaCtx)}};
    return std::nullopt;
  }();

  auto iterator = [&]() -> std::optional<Iterator> {
    if (t2)
      return makeIterator(*t2, semaCtx);
    return std::nullopt;
  }();

  return From{{/*Expectation=*/maybeApplyToV(convert, t0),
               /*Mappers=*/std::move(mappers),
               /*Iterator=*/std::move(iterator),
               /*LocatorList=*/makeObjects(t3, semaCtx)}};
}

// Full: empty

Grainsize make(const parser::OmpClause::Grainsize &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpGrainsizeClause
  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 = semantics::OmpGetUniqueModifier<parser::OmpPrescriptiveness>(mods);
  auto &t1 = std::get<parser::ScalarIntExpr>(inp.v.t);
  return Grainsize{
      {/*Prescriptiveness=*/maybeApplyToV(makePrescriptiveness, m0),
       /*Grainsize=*/makeExpr(t1, semaCtx)}};
}

HasDeviceAddr make(const parser::OmpClause::HasDeviceAddr &inp,
                   semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return HasDeviceAddr{/*List=*/makeObjects(inp.v, semaCtx)};
}

Hint make(const parser::OmpClause::Hint &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpHintClause
  return Hint{/*HintExpr=*/makeExpr(inp.v.v, semaCtx)};
}

Holds make(const parser::OmpClause::Holds &inp,
           semantics::SemanticsContext &semaCtx) {
  llvm_unreachable("Unimplemented: holds");
}

If make(const parser::OmpClause::If &inp,
        semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpIfClause
  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 =
      semantics::OmpGetUniqueModifier<parser::OmpDirectiveNameModifier>(mods);
  auto &t1 = std::get<parser::ScalarLogicalExpr>(inp.v.t);
  return If{
      {/*DirectiveNameModifier=*/maybeApplyToV([](auto &&s) { return s; }, m0),
       /*IfExpression=*/makeExpr(t1, semaCtx)}};
}

// Inbranch: empty

Inclusive make(const parser::OmpClause::Inclusive &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Inclusive{makeObjects(/*List=*/inp.v, semaCtx)};
}

Indirect make(const parser::OmpClause::Indirect &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v.v -> std::optional<parser::ScalarLogicalExpr>
  return Indirect{maybeApply(makeExprFn(semaCtx), inp.v.v)};
}

Init make(const parser::OmpClause::Init &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp -> empty
  llvm_unreachable("Empty: init");
}

Initializer make(const parser::OmpClause::Initializer &inp,
                 semantics::SemanticsContext &semaCtx) {
  llvm_unreachable("Empty: initializer");
}

InReduction make(const parser::OmpClause::InReduction &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpInReductionClause
  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 =
      semantics::OmpGetUniqueModifier<parser::OmpReductionIdentifier>(mods);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);
  assert(m0 && "OmpReductionIdentifier is required");

  return InReduction{
      {/*ReductionIdentifiers=*/{makeReductionOperator(*m0, semaCtx)},
       /*List=*/makeObjects(t1, semaCtx)}};
}

IsDevicePtr make(const parser::OmpClause::IsDevicePtr &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return IsDevicePtr{/*List=*/makeObjects(inp.v, semaCtx)};
}

Lastprivate make(const parser::OmpClause::Lastprivate &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpLastprivateClause
  CLAUSET_ENUM_CONVERT( //
      convert, parser::OmpLastprivateModifier::Value,
      Lastprivate::LastprivateModifier,
      // clang-format off
      MS(Conditional, Conditional)
      // clang-format on
  );

  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 =
      semantics::OmpGetUniqueModifier<parser::OmpLastprivateModifier>(mods);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);

  return Lastprivate{{/*LastprivateModifier=*/maybeApplyToV(convert, m0),
                      /*List=*/makeObjects(t1, semaCtx)}};
}

Linear make(const parser::OmpClause::Linear &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpLinearClause
  CLAUSET_ENUM_CONVERT( //
      convert, parser::OmpLinearModifier::Value, Linear::LinearModifier,
      // clang-format off
      MS(Ref,  Ref)
      MS(Val,  Val)
      MS(Uval, Uval)
      // clang-format on
  );

  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 =
      semantics::OmpGetUniqueModifier<parser::OmpStepComplexModifier>(mods);
  auto *m1 =
      semantics::OmpGetUniqueModifier<parser::OmpStepSimpleModifier>(mods);
  assert((!m0 || !m1) && "Simple and complex modifiers both present");

  auto *m2 = semantics::OmpGetUniqueModifier<parser::OmpLinearModifier>(mods);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);

  auto &&maybeStep = m0   ? maybeApplyToV(makeExprFn(semaCtx), m0)
                     : m1 ? maybeApplyToV(makeExprFn(semaCtx), m1)
                          : std::optional<Linear::StepComplexModifier>{};

  return Linear{{/*StepComplexModifier=*/std::move(maybeStep),
                 /*LinearModifier=*/maybeApplyToV(convert, m2),
                 /*List=*/makeObjects(t1, semaCtx)}};
}

Link make(const parser::OmpClause::Link &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Link{/*List=*/makeObjects(inp.v, semaCtx)};
}

LoopRange make(const parser::OmpClause::Looprange &inp,
               semantics::SemanticsContext &semaCtx) {
  llvm_unreachable("Unimplemented: looprange");
}

Map make(const parser::OmpClause::Map &inp,
         semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpMapClause
  CLAUSET_ENUM_CONVERT( //
      convertMapType, parser::OmpMapType::Value, Map::MapType,
      // clang-format off
      MS(Alloc,   Storage)
      MS(Delete,  Storage)
      MS(Release, Storage)
      MS(Storage, Storage)
      MS(From,    From)
      MS(To,      To)
      MS(Tofrom,  Tofrom)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convertMapTypeMod, parser::OmpMapTypeModifier::Value,
      Map::MapTypeModifier,
      // clang-format off
      MS(Always,    Always)
      MS(Close,     Close)
      MS(Ompx_Hold, OmpxHold)
      MS(Present,   Present)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convertAttachMod, parser::OmpAttachModifier::Value, Map::AttachModifier,
      // clang-format off
      MS(Always,  Always)
      MS(Auto,    Auto)
      MS(Never,   Never)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convertRefMod, parser::OmpRefModifier::Value, Map::RefModifier,
      // clang-format off
      MS(Ref_Ptee,     RefPtee)
      MS(Ref_Ptr,      RefPtr)
      MS(Ref_Ptr_Ptee, RefPtrPtee)
      // clang-format on
  );

  // Treat always, close, present, self, delete modifiers as map-type-
  // modifiers.
  auto &mods = semantics::OmpGetModifiers(inp.v);

  auto *t1 = semantics::OmpGetUniqueModifier<parser::OmpMapType>(mods);
  auto &t2 = std::get<parser::OmpObjectList>(inp.v.t);

  auto type = [&]() -> std::optional<Map::MapType> {
    if (t1)
      return convertMapType(t1->v);
    return std::nullopt;
  }();

  llvm::DenseSet<Map::MapTypeModifier> modSet;
  if (t1 && t1->v == parser::OmpMapType::Value::Delete)
    modSet.insert(Map::MapTypeModifier::Delete);

  for (auto *typeMod :
       semantics::OmpGetRepeatableModifier<parser::OmpMapTypeModifier>(mods)) {
    modSet.insert(convertMapTypeMod(typeMod->v));
  }
  if (semantics::OmpGetUniqueModifier<parser::OmpAlwaysModifier>(mods))
    modSet.insert(Map::MapTypeModifier::Always);
  if (semantics::OmpGetUniqueModifier<parser::OmpCloseModifier>(mods))
    modSet.insert(Map::MapTypeModifier::Close);
  if (semantics::OmpGetUniqueModifier<parser::OmpDeleteModifier>(mods))
    modSet.insert(Map::MapTypeModifier::Delete);
  if (semantics::OmpGetUniqueModifier<parser::OmpPresentModifier>(mods))
    modSet.insert(Map::MapTypeModifier::Present);
  if (semantics::OmpGetUniqueModifier<parser::OmpSelfModifier>(mods))
    modSet.insert(Map::MapTypeModifier::Self);
  if (semantics::OmpGetUniqueModifier<parser::OmpxHoldModifier>(mods))
    modSet.insert(Map::MapTypeModifier::OmpxHold);

  std::optional<Map::MapTypeModifiers> maybeTypeMods{};
  if (!modSet.empty())
    maybeTypeMods = Map::MapTypeModifiers(modSet.begin(), modSet.end());

  auto attachMod = [&]() -> std::optional<Map::AttachModifier> {
    if (auto *t =
            semantics::OmpGetUniqueModifier<parser::OmpAttachModifier>(mods))
      return convertAttachMod(t->v);
    return std::nullopt;
  }();

  auto refMod = [&]() -> std::optional<Map::RefModifier> {
    if (auto *t = semantics::OmpGetUniqueModifier<parser::OmpRefModifier>(mods))
      return convertRefMod(t->v);
    return std::nullopt;
  }();

  auto mappers = [&]() -> std::optional<List<Mapper>> {
    if (auto *t = semantics::OmpGetUniqueModifier<parser::OmpMapper>(mods))
      return List<Mapper>{Mapper{makeObject(t->v, semaCtx)}};
    return std::nullopt;
  }();

  auto iterator = [&]() -> std::optional<Iterator> {
    if (auto *t = semantics::OmpGetUniqueModifier<parser::OmpIterator>(mods))
      return makeIterator(*t, semaCtx);
    return std::nullopt;
  }();

  return Map{{/*MapType=*/std::move(type),
              /*MapTypeModifiers=*/std::move(maybeTypeMods),
              /*AttachModifier=*/std::move(attachMod),
              /*RefModifier=*/std::move(refMod), /*Mapper=*/std::move(mappers),
              /*Iterator=*/std::move(iterator),
              /*LocatorList=*/makeObjects(t2, semaCtx)}};
}

Match make(const parser::OmpClause::Match &inp,
           semantics::SemanticsContext &semaCtx) {
  return Match{};
}

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

// NoOpenmp: empty
// NoOpenmpRoutines: empty
// NoOpenmpConstructs: empty
// NoParallelism: empty
// Notinbranch: empty

Novariants make(const parser::OmpClause::Novariants &inp,
                semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarLogicalExpr
  return Novariants{/*DoNotUseVariant=*/makeExpr(inp.v, semaCtx)};
}

// Nowait: empty

NumTasks make(const parser::OmpClause::NumTasks &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpNumTasksClause
  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 = semantics::OmpGetUniqueModifier<parser::OmpPrescriptiveness>(mods);
  auto &t1 = std::get<parser::ScalarIntExpr>(inp.v.t);
  return NumTasks{{/*Prescriptiveness=*/maybeApplyToV(makePrescriptiveness, m0),
                   /*NumTasks=*/makeExpr(t1, semaCtx)}};
}

NumTeams make(const parser::OmpClause::NumTeams &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  List<NumTeams::Range> v{{{/*LowerBound=*/std::nullopt,
                            /*UpperBound=*/makeExpr(inp.v, semaCtx)}}};
  return NumTeams{/*List=*/v};
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
      convert1, parser::OmpOrderModifier::Value, Order::OrderModifier,
      // clang-format off
      MS(Reproducible,   Reproducible)
      MS(Unconstrained,  Unconstrained)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convert2, wrapped::Ordering, Order::Ordering,
      // clang-format off
      MS(Concurrent, Concurrent)
      // clang-format on
  );

  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *t0 = semantics::OmpGetUniqueModifier<parser::OmpOrderModifier>(mods);
  auto &t1 = std::get<wrapped::Ordering>(inp.v.t);

  return Order{{/*OrderModifier=*/maybeApplyToV(convert1, t0),
                /*Ordering=*/convert2(t1)}};
}

Ordered make(const parser::OmpClause::Ordered &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::ScalarIntConstantExpr>
  return Ordered{/*N=*/maybeApply(makeExprFn(semaCtx), inp.v)};
}

// See also Default.
Otherwise make(const parser::OmpClause::Otherwise &inp,
               semantics::SemanticsContext &semaCtx) {
  return Otherwise{};
}

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
      convert, wrapped::AffinityPolicy, ProcBind::AffinityPolicy,
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
  CLAUSET_ENUM_CONVERT( //
      convert, parser::OmpReductionModifier::Value,
      Reduction::ReductionModifier,
      // clang-format off
      MS(Inscan,  Inscan)
      MS(Task,    Task)
      MS(Default, Default)
      // clang-format on
  );

  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 =
      semantics::OmpGetUniqueModifier<parser::OmpReductionModifier>(mods);
  auto *m1 =
      semantics::OmpGetUniqueModifier<parser::OmpReductionIdentifier>(mods);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);
  assert(m1 && "OmpReductionIdentifier is required");

  return Reduction{
      {/*ReductionModifier=*/maybeApplyToV(convert, m0),
       /*ReductionIdentifiers=*/{makeReductionOperator(*m1, semaCtx)},
       /*List=*/makeObjects(t1, semaCtx)}};
}

// Relaxed: empty
// Release: empty

ReverseOffload make(const parser::OmpClause::ReverseOffload &inp,
                    semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::OmpReverseOffloadClause>
  auto &&maybeRequired = maybeApply(
      [&](const parser::OmpReverseOffloadClause &c) {
        return makeExpr(c.v, semaCtx);
      },
      inp.v);

  return ReverseOffload{/*Required=*/std::move(maybeRequired)};
}

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
      convert1, wrapped::Kind, Schedule::Kind,
      // clang-format off
      MS(Static,   Static)
      MS(Dynamic,  Dynamic)
      MS(Guided,   Guided)
      MS(Auto,     Auto)
      MS(Runtime,  Runtime)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convert2, parser::OmpOrderingModifier::Value, Schedule::OrderingModifier,
      // clang-format off
      MS(Monotonic,    Monotonic)
      MS(Nonmonotonic, Nonmonotonic)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convert3, parser::OmpChunkModifier::Value, Schedule::ChunkModifier,
      // clang-format off
      MS(Simd, Simd)
      // clang-format on
  );

  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *t0 = semantics::OmpGetUniqueModifier<parser::OmpOrderingModifier>(mods);
  auto *t1 = semantics::OmpGetUniqueModifier<parser::OmpChunkModifier>(mods);
  auto &t2 = std::get<wrapped::Kind>(inp.v.t);
  auto &t3 = std::get<std::optional<parser::ScalarIntExpr>>(inp.v.t);

  return Schedule{{/*Kind=*/convert1(t2),
                   /*OrderingModifier=*/maybeApplyToV(convert2, t0),
                   /*ChunkModifier=*/maybeApplyToV(convert3, t1),
                   /*ChunkSize=*/maybeApply(makeExprFn(semaCtx), t3)}};
}

// SeqCst: empty

SelfMaps make(const parser::OmpClause::SelfMaps &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::OmpSelfMapsClause>
  auto &&maybeRequired = maybeApply(
      [&](const parser::OmpSelfMapsClause &c) {
        return makeExpr(c.v, semaCtx);
      },
      inp.v);

  return SelfMaps{/*Required=*/std::move(maybeRequired)};
}

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

Permutation make(const parser::OmpClause::Permutation &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::list<parser::ScalarIntConstantExpr>
  return Permutation{/*ArgList=*/makeList(inp.v, makeExprFn(semaCtx))};
}

TaskReduction make(const parser::OmpClause::TaskReduction &inp,
                   semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpReductionClause
  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *m0 =
      semantics::OmpGetUniqueModifier<parser::OmpReductionIdentifier>(mods);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);
  assert(m0 && "OmpReductionIdentifier is required");

  return TaskReduction{
      {/*ReductionIdentifiers=*/{makeReductionOperator(*m0, semaCtx)},
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
  // inp.v -> parser::OmpToClause
  CLAUSET_ENUM_CONVERT( //
      convert, parser::OmpExpectation::Value, To::Expectation,
      // clang-format off
      MS(Present, Present)
      // clang-format on
  );

  auto &mods = semantics::OmpGetModifiers(inp.v);
  auto *t0 = semantics::OmpGetUniqueModifier<parser::OmpExpectation>(mods);
  auto *t1 = semantics::OmpGetUniqueModifier<parser::OmpMapper>(mods);
  auto *t2 = semantics::OmpGetUniqueModifier<parser::OmpIterator>(mods);
  auto &t3 = std::get<parser::OmpObjectList>(inp.v.t);

  auto mappers = [&]() -> std::optional<List<Mapper>> {
    if (t1)
      return List<Mapper>{Mapper{makeObject(t1->v, semaCtx)}};
    return std::nullopt;
  }();

  auto iterator = [&]() -> std::optional<Iterator> {
    if (t2)
      return makeIterator(*t2, semaCtx);
    return std::nullopt;
  }();

  return To{{/*Expectation=*/maybeApplyToV(convert, t0),
             /*Mappers=*/{std::move(mappers)},
             /*Iterator=*/std::move(iterator),
             /*LocatorList=*/makeObjects(t3, semaCtx)}};
}

UnifiedAddress make(const parser::OmpClause::UnifiedAddress &inp,
                    semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::OmpUnifiedAddressClause>
  auto &&maybeRequired = maybeApply(
      [&](const parser::OmpUnifiedAddressClause &c) {
        return makeExpr(c.v, semaCtx);
      },
      inp.v);

  return UnifiedAddress{/*Required=*/std::move(maybeRequired)};
}

UnifiedSharedMemory make(const parser::OmpClause::UnifiedSharedMemory &inp,
                         semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::OmpUnifiedSharedMemoryClause>
  auto &&maybeRequired = maybeApply(
      [&](const parser::OmpUnifiedSharedMemoryClause &c) {
        return makeExpr(c.v, semaCtx);
      },
      inp.v);

  return UnifiedSharedMemory{/*Required=*/std::move(maybeRequired)};
}

Uniform make(const parser::OmpClause::Uniform &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::list<parser::Name>
  return Uniform{/*ParameterList=*/makeList(inp.v, makeObjectFn(semaCtx))};
}

// Unknown: empty
// Untied: empty

Update make(const parser::OmpClause::Update &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpUpdateClause
  if (inp.v) {
    return common::visit(
        [](auto &&s) { return Update{/*DependenceType=*/makeDepType(s)}; },
        inp.v->u);
  } else {
    return Update{/*DependenceType=*/std::nullopt};
  }
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

When make(const parser::OmpClause::When &inp,
          semantics::SemanticsContext &semaCtx) {
  return When{};
}

// Write: empty
} // namespace clause

Clause makeClause(const parser::OmpClause &cls,
                  semantics::SemanticsContext &semaCtx) {
  return Fortran::common::visit( //
      common::visitors{
          [&](const parser::OmpClause::Default &s) {
            using DSA = parser::OmpDefaultClause::DataSharingAttribute;
            if (std::holds_alternative<DSA>(s.v.u)) {
              return makeClause(llvm::omp::Clause::OMPC_default,
                                clause::makeDefault(s, semaCtx), cls.source);
            } else {
              return makeClause(llvm::omp::Clause::OMPC_otherwise,
                                clause::makeOtherwise(s, semaCtx), cls.source);
            }
          },
          [&](auto &&s) {
            return makeClause(cls.Id(), clause::make(s, semaCtx), cls.source);
          },
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
