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
  return std::visit([](auto &&s) { return detail::getClauseIdForClass(s); },
                    clause.u);
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
    return std::visit([](auto &&s) { return visit(s); }, e.u);
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
  return std::visit(
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

std::optional<Object>
getBaseObject(const Object &object,
              Fortran::semantics::SemanticsContext &semaCtx) {
  // If it's just the symbol, then there is no base.
  if (!object.id())
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

namespace clause {
// Helper objects
#ifdef EMPTY_CLASS
#undef EMPTY_CLASS
#endif
#define EMPTY_CLASS(cls)                                                       \
  cls make(const parser::OmpClause::cls &, semantics::SemanticsContext &) {    \
    return cls{};                                                              \
  }                                                                            \
  [[maybe_unused]] extern int xyzzy_semicolon_absorber

#ifdef WRAPPER_CLASS
#undef WRAPPER_CLASS
#endif
#define WRAPPER_CLASS(cls, content)                                            \
  [[maybe_unused]] extern int xyzzy_semicolon_absorber
#define GEN_FLANG_CLAUSE_PARSER_CLASSES
#include "llvm/Frontend/OpenMP/OMP.inc"
#undef EMPTY_CLASS
#undef WRAPPER_CLASS

DefinedOperator makeDefinedOperator(const parser::DefinedOperator &inp,
                                    semantics::SemanticsContext &semaCtx) {
  return std::visit(
      common::visitors{
          [&](const parser::DefinedOpName &s) {
            return DefinedOperator{
                DefinedOperator::DefinedOpName{makeObject(s.v, semaCtx)}};
          },
          [&](const parser::DefinedOperator::IntrinsicOperator &s) {
            return DefinedOperator{s};
          },
      },
      inp.u);
}

ProcedureDesignator
makeProcedureDesignator(const parser::ProcedureDesignator &inp,
                        semantics::SemanticsContext &semaCtx) {
  return ProcedureDesignator{std::visit(
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
  return std::visit(
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

// Actual clauses. Each T (where OmpClause::T exists) has its "make".
Aligned make(const parser::OmpClause::Aligned &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpAlignedClause
  auto &t0 = std::get<parser::OmpObjectList>(inp.v.t);
  auto &t1 = std::get<std::optional<parser::ScalarIntConstantExpr>>(inp.v.t);

  return Aligned{{
      makeList(t0, semaCtx),
      maybeApply(makeExprFn(semaCtx), t1),
  }};
}

Allocate make(const parser::OmpClause::Allocate &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpAllocateClause
  using wrapped = parser::OmpAllocateClause;
  auto &t0 = std::get<std::optional<wrapped::AllocateModifier>>(inp.v.t);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);

  auto convert = [&](auto &&s) -> Allocate::Modifier {
    using Modifier = Allocate::Modifier;
    using Allocator = Modifier::Allocator;
    using Align = Modifier::Align;
    using ComplexModifier = Modifier::ComplexModifier;

    return std::visit(
        common::visitors{
            [&](const wrapped::AllocateModifier::Allocator &v) {
              return Modifier{Allocator{makeExpr(v.v, semaCtx)}};
            },
            [&](const wrapped::AllocateModifier::ComplexModifier &v) {
              auto &s0 = std::get<wrapped::AllocateModifier::Allocator>(v.t);
              auto &s1 = std::get<wrapped::AllocateModifier::Align>(v.t);
              return Modifier{ComplexModifier{{
                  Allocator{makeExpr(s0.v, semaCtx)},
                  Align{makeExpr(s1.v, semaCtx)},
              }}};
            },
            [&](const wrapped::AllocateModifier::Align &v) {
              return Modifier{Align{makeExpr(v.v, semaCtx)}};
            },
        },
        s.u);
  };

  return Allocate{{maybeApply(convert, t0), makeList(t1, semaCtx)}};
}

Allocator make(const parser::OmpClause::Allocator &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return Allocator{makeExpr(inp.v, semaCtx)};
}

// Never called, but needed for using "make" as a Clause visitor.
// See comment about "requires" clauses in Clauses.h.
AtomicDefaultMemOrder make(const parser::OmpClause::AtomicDefaultMemOrder &inp,
                           semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpAtomicDefaultMemOrderClause
  return AtomicDefaultMemOrder{inp.v.v};
}

Collapse make(const parser::OmpClause::Collapse &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntConstantExpr
  return Collapse{makeExpr(inp.v, semaCtx)};
}

Copyin make(const parser::OmpClause::Copyin &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Copyin{makeList(inp.v, semaCtx)};
}

Copyprivate make(const parser::OmpClause::Copyprivate &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Copyprivate{makeList(inp.v, semaCtx)};
}

Defaultmap make(const parser::OmpClause::Defaultmap &inp,
                semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDefaultmapClause
  using wrapped = parser::OmpDefaultmapClause;

  auto &t0 = std::get<wrapped::ImplicitBehavior>(inp.v.t);
  auto &t1 = std::get<std::optional<wrapped::VariableCategory>>(inp.v.t);
  return Defaultmap{{t0, t1}};
}

Default make(const parser::OmpClause::Default &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDefaultClause
  return Default{inp.v.v};
}

Depend make(const parser::OmpClause::Depend &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDependClause
  using wrapped = parser::OmpDependClause;

  return std::visit(
      common::visitors{
          [&](const wrapped::Source &s) { return Depend{Depend::Source{}}; },
          [&](const wrapped::Sink &s) {
            auto convert = [&](const parser::OmpDependSinkVec &v) {
              auto &t0 = std::get<parser::Name>(v.t);
              auto &t1 =
                  std::get<std::optional<parser::OmpDependSinkVecLength>>(v.t);
              auto convert1 = [&](const parser::OmpDependSinkVecLength &u) {
                auto &s0 = std::get<parser::DefinedOperator>(u.t);
                auto &s1 = std::get<parser::ScalarIntConstantExpr>(u.t);
                return Depend::Sink::Length{makeDefinedOperator(s0, semaCtx),
                                            makeExpr(s1, semaCtx)};
              };
              return Depend::Sink::Vec{makeObject(t0, semaCtx),
                                       maybeApply(convert1, t1)};
            };
            return Depend{Depend::Sink{makeList(s.v, convert)}};
          },
          [&](const wrapped::InOut &s) {
            auto &t0 = std::get<parser::OmpDependenceType>(s.t);
            auto &t1 = std::get<std::list<parser::Designator>>(s.t);
            auto convert = [&](const parser::Designator &t) {
              return makeObject(t, semaCtx);
            };
            return Depend{Depend::InOut{{t0.v, makeList(t1, convert)}}};
          },
      },
      inp.v.u);
}

Device make(const parser::OmpClause::Device &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDeviceClause
  using wrapped = parser::OmpDeviceClause;

  auto &t0 = std::get<std::optional<wrapped::DeviceModifier>>(inp.v.t);
  auto &t1 = std::get<parser::ScalarIntExpr>(inp.v.t);
  return Device{{t0, makeExpr(t1, semaCtx)}};
}

DeviceType make(const parser::OmpClause::DeviceType &inp,
                semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpDeviceTypeClause
  return DeviceType{inp.v.v};
}

DistSchedule make(const parser::OmpClause::DistSchedule &inp,
                  semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::ScalarIntExpr>
  return DistSchedule{maybeApply(makeExprFn(semaCtx), inp.v)};
}

Enter make(const parser::OmpClause::Enter &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Enter{makeList(inp.v, semaCtx)};
}

Filter make(const parser::OmpClause::Filter &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return Filter{makeExpr(inp.v, semaCtx)};
}

Final make(const parser::OmpClause::Final &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarLogicalExpr
  return Final{makeExpr(inp.v, semaCtx)};
}

Firstprivate make(const parser::OmpClause::Firstprivate &inp,
                  semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Firstprivate{makeList(inp.v, semaCtx)};
}

From make(const parser::OmpClause::From &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return From{makeList(inp.v, semaCtx)};
}

Grainsize make(const parser::OmpClause::Grainsize &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return Grainsize{makeExpr(inp.v, semaCtx)};
}

HasDeviceAddr make(const parser::OmpClause::HasDeviceAddr &inp,
                   semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return HasDeviceAddr{makeList(inp.v, semaCtx)};
}

Hint make(const parser::OmpClause::Hint &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ConstantExpr
  return Hint{makeExpr(inp.v, semaCtx)};
}

If make(const parser::OmpClause::If &inp,
        semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpIfClause
  using wrapped = parser::OmpIfClause;

  auto &t0 = std::get<std::optional<wrapped::DirectiveNameModifier>>(inp.v.t);
  auto &t1 = std::get<parser::ScalarLogicalExpr>(inp.v.t);
  return If{{t0, makeExpr(t1, semaCtx)}};
}

InReduction make(const parser::OmpClause::InReduction &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpInReductionClause
  auto &t0 = std::get<parser::OmpReductionOperator>(inp.v.t);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);
  return InReduction{
      {makeReductionOperator(t0, semaCtx), makeList(t1, semaCtx)}};
}

IsDevicePtr make(const parser::OmpClause::IsDevicePtr &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return IsDevicePtr{makeList(inp.v, semaCtx)};
}

Lastprivate make(const parser::OmpClause::Lastprivate &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Lastprivate{makeList(inp.v, semaCtx)};
}

Linear make(const parser::OmpClause::Linear &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpLinearClause
  using wrapped = parser::OmpLinearClause;

  return std::visit(
      common::visitors{
          [&](const wrapped::WithModifier &s) {
            return Linear{{Linear::Modifier{s.modifier.v},
                           makeList(s.names, makeObjectFn(semaCtx)),
                           maybeApply(makeExprFn(semaCtx), s.step)}};
          },
          [&](const wrapped::WithoutModifier &s) {
            return Linear{{std::nullopt,
                           makeList(s.names, makeObjectFn(semaCtx)),
                           maybeApply(makeExprFn(semaCtx), s.step)}};
          },
      },
      inp.v.u);
}

Link make(const parser::OmpClause::Link &inp,
          semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Link{makeList(inp.v, semaCtx)};
}

Map make(const parser::OmpClause::Map &inp,
         semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpMapClause
  auto &t0 = std::get<std::optional<parser::OmpMapType>>(inp.v.t);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);
  auto convert = [](const parser::OmpMapType &s) {
    auto &s0 = std::get<std::optional<parser::OmpMapType::Always>>(s.t);
    auto &s1 = std::get<parser::OmpMapType::Type>(s.t);
    auto convertT = [](parser::OmpMapType::Always) {
      return Map::MapType::Always{};
    };
    return Map::MapType{{maybeApply(convertT, s0), s1}};
  };
  return Map{{maybeApply(convert, t0), makeList(t1, semaCtx)}};
}

Nocontext make(const parser::OmpClause::Nocontext &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarLogicalExpr
  return Nocontext{makeExpr(inp.v, semaCtx)};
}

Nontemporal make(const parser::OmpClause::Nontemporal &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::list<parser::Name>
  return Nontemporal{makeList(inp.v, makeObjectFn(semaCtx))};
}

Novariants make(const parser::OmpClause::Novariants &inp,
                semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarLogicalExpr
  return Novariants{makeExpr(inp.v, semaCtx)};
}

NumTasks make(const parser::OmpClause::NumTasks &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return NumTasks{makeExpr(inp.v, semaCtx)};
}

NumTeams make(const parser::OmpClause::NumTeams &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return NumTeams{makeExpr(inp.v, semaCtx)};
}

NumThreads make(const parser::OmpClause::NumThreads &inp,
                semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return NumThreads{makeExpr(inp.v, semaCtx)};
}

OmpxDynCgroupMem make(const parser::OmpClause::OmpxDynCgroupMem &inp,
                      semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return OmpxDynCgroupMem{makeExpr(inp.v, semaCtx)};
}

Ordered make(const parser::OmpClause::Ordered &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::ScalarIntConstantExpr>
  return Ordered{maybeApply(makeExprFn(semaCtx), inp.v)};
}

Order make(const parser::OmpClause::Order &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpOrderClause
  using wrapped = parser::OmpOrderClause;
  auto &t0 = std::get<std::optional<parser::OmpOrderModifier>>(inp.v.t);
  auto &t1 = std::get<wrapped::Type>(inp.v.t);
  auto convert = [](const parser::OmpOrderModifier &s) -> Order::Kind {
    return std::get<parser::OmpOrderModifier::Kind>(s.u);
  };
  return Order{{maybeApply(convert, t0), t1}};
}

Partial make(const parser::OmpClause::Partial &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::optional<parser::ScalarIntConstantExpr>
  return Partial{maybeApply(makeExprFn(semaCtx), inp.v)};
}

Priority make(const parser::OmpClause::Priority &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return Priority{makeExpr(inp.v, semaCtx)};
}

Private make(const parser::OmpClause::Private &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Private{makeList(inp.v, semaCtx)};
}

ProcBind make(const parser::OmpClause::ProcBind &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpProcBindClause
  return ProcBind{inp.v.v};
}

Reduction make(const parser::OmpClause::Reduction &inp,
               semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpReductionClause
  auto &t0 = std::get<parser::OmpReductionOperator>(inp.v.t);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);
  return Reduction{{makeReductionOperator(t0, semaCtx), makeList(t1, semaCtx)}};
}

Safelen make(const parser::OmpClause::Safelen &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntConstantExpr
  return Safelen{makeExpr(inp.v, semaCtx)};
}

Schedule make(const parser::OmpClause::Schedule &inp,
              semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpScheduleClause
  using wrapped = parser::OmpScheduleClause;

  auto &t0 = std::get<std::optional<parser::OmpScheduleModifier>>(inp.v.t);
  auto &t1 = std::get<wrapped::ScheduleType>(inp.v.t);
  auto &t2 = std::get<std::optional<parser::ScalarIntExpr>>(inp.v.t);

  auto convert = [](auto &&s) -> Schedule::ScheduleModifier {
    auto &s0 = std::get<parser::OmpScheduleModifier::Modifier1>(s.t);
    auto &s1 =
        std::get<std::optional<parser::OmpScheduleModifier::Modifier2>>(s.t);

    auto convert1 = [](auto &&v) { // Modifier1 or Modifier2
      return v.v.v;
    };
    return Schedule::ScheduleModifier{{s0.v.v, maybeApply(convert1, s1)}};
  };

  return Schedule{
      {maybeApply(convert, t0), t1, maybeApply(makeExprFn(semaCtx), t2)}};
}

Shared make(const parser::OmpClause::Shared &inp,
            semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return Shared{makeList(inp.v, semaCtx)};
}

Simdlen make(const parser::OmpClause::Simdlen &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntConstantExpr
  return Simdlen{makeExpr(inp.v, semaCtx)};
}

Sizes make(const parser::OmpClause::Sizes &inp,
           semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::list<parser::ScalarIntExpr>
  return Sizes{makeList(inp.v, makeExprFn(semaCtx))};
}

TaskReduction make(const parser::OmpClause::TaskReduction &inp,
                   semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpReductionClause
  auto &t0 = std::get<parser::OmpReductionOperator>(inp.v.t);
  auto &t1 = std::get<parser::OmpObjectList>(inp.v.t);
  return TaskReduction{
      {makeReductionOperator(t0, semaCtx), makeList(t1, semaCtx)}};
}

ThreadLimit make(const parser::OmpClause::ThreadLimit &inp,
                 semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::ScalarIntExpr
  return ThreadLimit{makeExpr(inp.v, semaCtx)};
}

To make(const parser::OmpClause::To &inp,
        semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return To{makeList(inp.v, semaCtx)};
}

Uniform make(const parser::OmpClause::Uniform &inp,
             semantics::SemanticsContext &semaCtx) {
  // inp.v -> std::list<parser::Name>
  return Uniform{makeList(inp.v, makeObjectFn(semaCtx))};
}

UseDeviceAddr make(const parser::OmpClause::UseDeviceAddr &inp,
                   semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return UseDeviceAddr{makeList(inp.v, semaCtx)};
}

UseDevicePtr make(const parser::OmpClause::UseDevicePtr &inp,
                  semantics::SemanticsContext &semaCtx) {
  // inp.v -> parser::OmpObjectList
  return UseDevicePtr{makeList(inp.v, semaCtx)};
}
} // namespace clause

Clause makeClause(const Fortran::parser::OmpClause &cls,
                  semantics::SemanticsContext &semaCtx) {
  return std::visit(
      [&](auto &&s) {
        return makeClause(getClauseId(cls), clause::make(s, semaCtx),
                          cls.source);
      },
      cls.u);
}

List<Clause> makeList(const parser::OmpClauseList &clauses,
                      semantics::SemanticsContext &semaCtx) {
  return makeList(clauses.v, [&](const parser::OmpClause &s) {
    return makeClause(s, semaCtx);
  });
}
} // namespace Fortran::lower::omp
