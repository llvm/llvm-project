//===-- Clauses.h -- OpenMP clause handling -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_LOWER_OPENMP_CLAUSES_H
#define FORTRAN_LOWER_OPENMP_CLAUSES_H

#include "ClauseT.h"

#include "flang/Evaluate/expression.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <type_traits>
#include <utility>

namespace Fortran::lower::omp {
using namespace Fortran;
using SomeType = evaluate::SomeType;
using SomeExpr = semantics::SomeExpr;
using MaybeExpr = semantics::MaybeExpr;

using SymIdent = semantics::Symbol *;
using SymReference = SomeExpr;

template <typename T>
using List = tomp::ListT<T>;
} // namespace Fortran::lower::omp

namespace tomp {
template <>
struct ObjectT<Fortran::lower::omp::SymIdent,
               Fortran::lower::omp::SymReference> {
  using IdType = Fortran::lower::omp::SymIdent;
  using ExprType = Fortran::lower::omp::SymReference;

  const IdType &id() const { return symbol; }
  const std::optional<ExprType> &ref() const { return designator; }

  IdType symbol;
  std::optional<ExprType> designator;
};
} // namespace tomp

namespace Fortran::lower::omp {

using Object = tomp::ObjectT<SymIdent, SymReference>;
using ObjectList = tomp::ObjectListT<SymIdent, SymReference>;

Object makeObject(const parser::OmpObject &object,
                  semantics::SemanticsContext &semaCtx);
Object makeObject(const parser::Name &name,
                  semantics::SemanticsContext &semaCtx);
Object makeObject(const parser::Designator &dsg,
                  semantics::SemanticsContext &semaCtx);
Object makeObject(const parser::StructureComponent &comp,
                  semantics::SemanticsContext &semaCtx);

inline auto makeObjectFn(semantics::SemanticsContext &semaCtx) {
  return [&](auto &&s) { return makeObject(s, semaCtx); };
}

template <typename T>
SomeExpr makeExpr(T &&pftExpr, semantics::SemanticsContext &semaCtx) {
  auto maybeExpr = evaluate::ExpressionAnalyzer(semaCtx).Analyze(pftExpr);
  assert(maybeExpr);
  return std::move(*maybeExpr);
}

inline auto makeExprFn(semantics::SemanticsContext &semaCtx) {
  return [&](auto &&s) { return makeExpr(s, semaCtx); };
}

template <
    typename ContainerTy, typename FunctionTy,
    typename ElemTy = typename llvm::remove_cvref_t<ContainerTy>::value_type,
    typename ResultTy = std::invoke_result_t<FunctionTy, ElemTy>>
List<ResultTy> makeList(ContainerTy &&container, FunctionTy &&func) {
  List<ResultTy> v;
  llvm::transform(container, std::back_inserter(v), func);
  return v;
}

inline ObjectList makeList(const parser::OmpObjectList &objects,
                           semantics::SemanticsContext &semaCtx) {
  return makeList(objects.v, makeObjectFn(semaCtx));
}

template <typename FuncTy, typename ElemTy,
          typename ResultTy = std::invoke_result_t<FuncTy, ElemTy>>
std::optional<ResultTy> maybeApply(FuncTy &&func,
                                   const std::optional<ElemTy> &inp) {
  if (!inp)
    return std::nullopt;
  return std::move(func(*inp));
}

std::optional<Object>
getBaseObject(const Object &object,
              Fortran::semantics::SemanticsContext &semaCtx);

namespace clause {
using DefinedOperator = tomp::clause::DefinedOperatorT<SymIdent, SymReference>;
using ProcedureDesignator =
    tomp::clause::ProcedureDesignatorT<SymIdent, SymReference>;
using ReductionOperator =
    tomp::clause::ReductionOperatorT<SymIdent, SymReference>;

#ifdef EMPTY_CLASS
#undef EMPTY_CLASS
#endif
#define EMPTY_CLASS(cls)                                                       \
  using cls = tomp::clause::cls##T<SymIdent, SymReference>

#ifdef WRAPPER_CLASS
#undef WRAPPER_CLASS
#endif
#define WRAPPER_CLASS(cls, content)                                            \
  [[maybe_unused]] extern int xyzzy_semicolon_absorber
#define GEN_FLANG_CLAUSE_PARSER_CLASSES
#include "llvm/Frontend/OpenMP/OMP.inc"
#undef EMPTY_CLASS
#undef WRAPPER_CLASS

// "Requires" clauses are handled early on, and the aggregated information
// is stored in the Symbol details of modules, programs, and subprograms.
// These clauses are still handled here to cover all alternatives in the
// main clause variant.

using Aligned = tomp::clause::AlignedT<SymIdent, SymReference>;
using Allocate = tomp::clause::AllocateT<SymIdent, SymReference>;
using Allocator = tomp::clause::AllocatorT<SymIdent, SymReference>;
using AtomicDefaultMemOrder =
    tomp::clause::AtomicDefaultMemOrderT<SymIdent, SymReference>;
using Collapse = tomp::clause::CollapseT<SymIdent, SymReference>;
using Copyin = tomp::clause::CopyinT<SymIdent, SymReference>;
using Copyprivate = tomp::clause::CopyprivateT<SymIdent, SymReference>;
using Defaultmap = tomp::clause::DefaultmapT<SymIdent, SymReference>;
using Default = tomp::clause::DefaultT<SymIdent, SymReference>;
using Depend = tomp::clause::DependT<SymIdent, SymReference>;
using Device = tomp::clause::DeviceT<SymIdent, SymReference>;
using DeviceType = tomp::clause::DeviceTypeT<SymIdent, SymReference>;
using DistSchedule = tomp::clause::DistScheduleT<SymIdent, SymReference>;
using Enter = tomp::clause::EnterT<SymIdent, SymReference>;
using Filter = tomp::clause::FilterT<SymIdent, SymReference>;
using Final = tomp::clause::FinalT<SymIdent, SymReference>;
using Firstprivate = tomp::clause::FirstprivateT<SymIdent, SymReference>;
using From = tomp::clause::FromT<SymIdent, SymReference>;
using Grainsize = tomp::clause::GrainsizeT<SymIdent, SymReference>;
using HasDeviceAddr = tomp::clause::HasDeviceAddrT<SymIdent, SymReference>;
using Hint = tomp::clause::HintT<SymIdent, SymReference>;
using If = tomp::clause::IfT<SymIdent, SymReference>;
using InReduction = tomp::clause::InReductionT<SymIdent, SymReference>;
using IsDevicePtr = tomp::clause::IsDevicePtrT<SymIdent, SymReference>;
using Lastprivate = tomp::clause::LastprivateT<SymIdent, SymReference>;
using Linear = tomp::clause::LinearT<SymIdent, SymReference>;
using Link = tomp::clause::LinkT<SymIdent, SymReference>;
using Map = tomp::clause::MapT<SymIdent, SymReference>;
using Nocontext = tomp::clause::NocontextT<SymIdent, SymReference>;
using Nontemporal = tomp::clause::NontemporalT<SymIdent, SymReference>;
using Novariants = tomp::clause::NovariantsT<SymIdent, SymReference>;
using NumTasks = tomp::clause::NumTasksT<SymIdent, SymReference>;
using NumTeams = tomp::clause::NumTeamsT<SymIdent, SymReference>;
using NumThreads = tomp::clause::NumThreadsT<SymIdent, SymReference>;
using OmpxDynCgroupMem =
    tomp::clause::OmpxDynCgroupMemT<SymIdent, SymReference>;
using Ordered = tomp::clause::OrderedT<SymIdent, SymReference>;
using Order = tomp::clause::OrderT<SymIdent, SymReference>;
using Partial = tomp::clause::PartialT<SymIdent, SymReference>;
using Priority = tomp::clause::PriorityT<SymIdent, SymReference>;
using Private = tomp::clause::PrivateT<SymIdent, SymReference>;
using ProcBind = tomp::clause::ProcBindT<SymIdent, SymReference>;
using Reduction = tomp::clause::ReductionT<SymIdent, SymReference>;
using Safelen = tomp::clause::SafelenT<SymIdent, SymReference>;
using Schedule = tomp::clause::ScheduleT<SymIdent, SymReference>;
using Shared = tomp::clause::SharedT<SymIdent, SymReference>;
using Simdlen = tomp::clause::SimdlenT<SymIdent, SymReference>;
using Sizes = tomp::clause::SizesT<SymIdent, SymReference>;
using TaskReduction = tomp::clause::TaskReductionT<SymIdent, SymReference>;
using ThreadLimit = tomp::clause::ThreadLimitT<SymIdent, SymReference>;
using To = tomp::clause::ToT<SymIdent, SymReference>;
using Uniform = tomp::clause::UniformT<SymIdent, SymReference>;
using UseDeviceAddr = tomp::clause::UseDeviceAddrT<SymIdent, SymReference>;
using UseDevicePtr = tomp::clause::UseDevicePtrT<SymIdent, SymReference>;
} // namespace clause

struct Clause : public tomp::ClauseT<SymIdent, SymReference> {
  parser::CharBlock source;
};

template <typename Specific>
Clause makeClause(llvm::omp::Clause id, Specific &&specific,
                  parser::CharBlock source = {}) {
  return Clause{{id, specific}, source};
}

Clause makeClause(const Fortran::parser::OmpClause &cls,
                  semantics::SemanticsContext &semaCtx);

List<Clause> makeList(const parser::OmpClauseList &clauses,
                      semantics::SemanticsContext &semaCtx);
} // namespace Fortran::lower::omp

#endif // FORTRAN_LOWER_OPENMP_CLAUSES_H
