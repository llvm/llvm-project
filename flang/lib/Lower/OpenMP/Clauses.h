//===-- Clauses.h -- OpenMP clause handling -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_LOWER_OPENMP_CLAUSES_H
#define FORTRAN_LOWER_OPENMP_CLAUSES_H

#include "flang/Evaluate/expression.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/OpenMP/ClauseT.h"

#include <optional>
#include <type_traits>
#include <utility>

namespace Fortran::lower::omp {
using namespace Fortran;
using SomeType = evaluate::SomeType;
using SomeExpr = semantics::SomeExpr;
using MaybeExpr = semantics::MaybeExpr;

using TypeTy = SomeType;
using IdentTy = semantics::Symbol *;
using ExprTy = SomeExpr;

template <typename T>
using List = tomp::ListT<T>;
} // namespace Fortran::lower::omp

namespace tomp::type {
template <>
struct ObjectT<Fortran::lower::omp::IdentTy, Fortran::lower::omp::ExprTy> {
  using IdType = Fortran::lower::omp::IdentTy;
  using ExprType = Fortran::lower::omp::ExprTy;

  const IdType &id() const { return symbol; }
  const std::optional<ExprType> &ref() const { return designator; }

  IdType symbol;
  std::optional<ExprType> designator;
};
} // namespace tomp::type

namespace Fortran::lower::omp {

using Object = tomp::ObjectT<IdentTy, ExprTy>;
using ObjectList = tomp::ObjectListT<IdentTy, ExprTy>;

Object makeObject(const parser::OmpObject &object,
                  semantics::SemanticsContext &semaCtx);
Object makeObject(const parser::Name &name,
                  semantics::SemanticsContext &semaCtx);
Object makeObject(const parser::Designator &dsg,
                  semantics::SemanticsContext &semaCtx);
Object makeObject(const parser::StructureComponent &comp,
                  semantics::SemanticsContext &semaCtx);

inline auto makeObjectF(semantics::SemanticsContext &semaCtx) {
  return [&](auto &&s) { return makeObject(s, semaCtx); };
}

template <typename T>
SomeExpr makeExpr(T &&inp, semantics::SemanticsContext &semaCtx) {
  auto maybeExpr = evaluate::ExpressionAnalyzer(semaCtx).Analyze(inp);
  assert(maybeExpr);
  return std::move(*maybeExpr);
}

inline auto makeExprF(semantics::SemanticsContext &semaCtx) {
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
  return makeList(objects.v, makeObjectF(semaCtx));
}

template <typename FuncTy, //
          typename ArgTy,  //
          typename ResultTy = std::invoke_result_t<FuncTy, ArgTy>>
std::optional<ResultTy> maybeApply(FuncTy &&func,
                                   const std::optional<ArgTy> &arg) {
  if (!arg)
    return std::nullopt;
  return std::move(func(*arg));
}

std::optional<Object>
getBaseObject(const Object &object,
              Fortran::semantics::SemanticsContext &semaCtx);

namespace clause {
using DefinedOperator = tomp::type::DefinedOperatorT<IdentTy, ExprTy>;
using ProcedureDesignator = tomp::type::ProcedureDesignatorT<IdentTy, ExprTy>;
using ReductionOperator = tomp::type::ReductionIdentifierT<IdentTy, ExprTy>;

using AcqRel = tomp::clause::AcqRelT<TypeTy, IdentTy, ExprTy>;
using Acquire = tomp::clause::AcquireT<TypeTy, IdentTy, ExprTy>;
using AdjustArgs = tomp::clause::AdjustArgsT<TypeTy, IdentTy, ExprTy>;
using Affinity = tomp::clause::AffinityT<TypeTy, IdentTy, ExprTy>;
using Aligned = tomp::clause::AlignedT<TypeTy, IdentTy, ExprTy>;
using Align = tomp::clause::AlignT<TypeTy, IdentTy, ExprTy>;
using Allocate = tomp::clause::AllocateT<TypeTy, IdentTy, ExprTy>;
using Allocator = tomp::clause::AllocatorT<TypeTy, IdentTy, ExprTy>;
using AppendArgs = tomp::clause::AppendArgsT<TypeTy, IdentTy, ExprTy>;
using AtomicDefaultMemOrder =
    tomp::clause::AtomicDefaultMemOrderT<TypeTy, IdentTy, ExprTy>;
using At = tomp::clause::AtT<TypeTy, IdentTy, ExprTy>;
using Bind = tomp::clause::BindT<TypeTy, IdentTy, ExprTy>;
using CancellationConstructType =
    tomp::clause::CancellationConstructTypeT<TypeTy, IdentTy, ExprTy>;
using Capture = tomp::clause::CaptureT<TypeTy, IdentTy, ExprTy>;
using Collapse = tomp::clause::CollapseT<TypeTy, IdentTy, ExprTy>;
using Compare = tomp::clause::CompareT<TypeTy, IdentTy, ExprTy>;
using Copyin = tomp::clause::CopyinT<TypeTy, IdentTy, ExprTy>;
using Copyprivate = tomp::clause::CopyprivateT<TypeTy, IdentTy, ExprTy>;
using Defaultmap = tomp::clause::DefaultmapT<TypeTy, IdentTy, ExprTy>;
using Default = tomp::clause::DefaultT<TypeTy, IdentTy, ExprTy>;
using Depend = tomp::clause::DependT<TypeTy, IdentTy, ExprTy>;
using Depobj = tomp::clause::DepobjT<TypeTy, IdentTy, ExprTy>;
using Destroy = tomp::clause::DestroyT<TypeTy, IdentTy, ExprTy>;
using Detach = tomp::clause::DetachT<TypeTy, IdentTy, ExprTy>;
using Device = tomp::clause::DeviceT<TypeTy, IdentTy, ExprTy>;
using DeviceType = tomp::clause::DeviceTypeT<TypeTy, IdentTy, ExprTy>;
using DistSchedule = tomp::clause::DistScheduleT<TypeTy, IdentTy, ExprTy>;
using Doacross = tomp::clause::DoacrossT<TypeTy, IdentTy, ExprTy>;
using DynamicAllocators =
    tomp::clause::DynamicAllocatorsT<TypeTy, IdentTy, ExprTy>;
using Enter = tomp::clause::EnterT<TypeTy, IdentTy, ExprTy>;
using Exclusive = tomp::clause::ExclusiveT<TypeTy, IdentTy, ExprTy>;
using Fail = tomp::clause::FailT<TypeTy, IdentTy, ExprTy>;
using Filter = tomp::clause::FilterT<TypeTy, IdentTy, ExprTy>;
using Final = tomp::clause::FinalT<TypeTy, IdentTy, ExprTy>;
using Firstprivate = tomp::clause::FirstprivateT<TypeTy, IdentTy, ExprTy>;
using Flush = tomp::clause::FlushT<TypeTy, IdentTy, ExprTy>;
using From = tomp::clause::FromT<TypeTy, IdentTy, ExprTy>;
using Full = tomp::clause::FullT<TypeTy, IdentTy, ExprTy>;
using Grainsize = tomp::clause::GrainsizeT<TypeTy, IdentTy, ExprTy>;
using HasDeviceAddr = tomp::clause::HasDeviceAddrT<TypeTy, IdentTy, ExprTy>;
using Hint = tomp::clause::HintT<TypeTy, IdentTy, ExprTy>;
using If = tomp::clause::IfT<TypeTy, IdentTy, ExprTy>;
using Inbranch = tomp::clause::InbranchT<TypeTy, IdentTy, ExprTy>;
using Inclusive = tomp::clause::InclusiveT<TypeTy, IdentTy, ExprTy>;
using Indirect = tomp::clause::IndirectT<TypeTy, IdentTy, ExprTy>;
using Init = tomp::clause::InitT<TypeTy, IdentTy, ExprTy>;
using InReduction = tomp::clause::InReductionT<TypeTy, IdentTy, ExprTy>;
using IsDevicePtr = tomp::clause::IsDevicePtrT<TypeTy, IdentTy, ExprTy>;
using Lastprivate = tomp::clause::LastprivateT<TypeTy, IdentTy, ExprTy>;
using Linear = tomp::clause::LinearT<TypeTy, IdentTy, ExprTy>;
using Link = tomp::clause::LinkT<TypeTy, IdentTy, ExprTy>;
using Map = tomp::clause::MapT<TypeTy, IdentTy, ExprTy>;
using Match = tomp::clause::MatchT<TypeTy, IdentTy, ExprTy>;
using MemoryOrder = tomp::clause::MemoryOrderT<TypeTy, IdentTy, ExprTy>;
using Mergeable = tomp::clause::MergeableT<TypeTy, IdentTy, ExprTy>;
using Message = tomp::clause::MessageT<TypeTy, IdentTy, ExprTy>;
using Nocontext = tomp::clause::NocontextT<TypeTy, IdentTy, ExprTy>;
using Nogroup = tomp::clause::NogroupT<TypeTy, IdentTy, ExprTy>;
using Nontemporal = tomp::clause::NontemporalT<TypeTy, IdentTy, ExprTy>;
using Notinbranch = tomp::clause::NotinbranchT<TypeTy, IdentTy, ExprTy>;
using Novariants = tomp::clause::NovariantsT<TypeTy, IdentTy, ExprTy>;
using Nowait = tomp::clause::NowaitT<TypeTy, IdentTy, ExprTy>;
using NumTasks = tomp::clause::NumTasksT<TypeTy, IdentTy, ExprTy>;
using NumTeams = tomp::clause::NumTeamsT<TypeTy, IdentTy, ExprTy>;
using NumThreads = tomp::clause::NumThreadsT<TypeTy, IdentTy, ExprTy>;
using OmpxAttribute = tomp::clause::OmpxAttributeT<TypeTy, IdentTy, ExprTy>;
using OmpxBare = tomp::clause::OmpxBareT<TypeTy, IdentTy, ExprTy>;
using OmpxDynCgroupMem =
    tomp::clause::OmpxDynCgroupMemT<TypeTy, IdentTy, ExprTy>;
using Ordered = tomp::clause::OrderedT<TypeTy, IdentTy, ExprTy>;
using Order = tomp::clause::OrderT<TypeTy, IdentTy, ExprTy>;
using Partial = tomp::clause::PartialT<TypeTy, IdentTy, ExprTy>;
using Priority = tomp::clause::PriorityT<TypeTy, IdentTy, ExprTy>;
using Private = tomp::clause::PrivateT<TypeTy, IdentTy, ExprTy>;
using ProcBind = tomp::clause::ProcBindT<TypeTy, IdentTy, ExprTy>;
using Read = tomp::clause::ReadT<TypeTy, IdentTy, ExprTy>;
using Reduction = tomp::clause::ReductionT<TypeTy, IdentTy, ExprTy>;
using Relaxed = tomp::clause::RelaxedT<TypeTy, IdentTy, ExprTy>;
using Release = tomp::clause::ReleaseT<TypeTy, IdentTy, ExprTy>;
using ReverseOffload = tomp::clause::ReverseOffloadT<TypeTy, IdentTy, ExprTy>;
using Safelen = tomp::clause::SafelenT<TypeTy, IdentTy, ExprTy>;
using Schedule = tomp::clause::ScheduleT<TypeTy, IdentTy, ExprTy>;
using SeqCst = tomp::clause::SeqCstT<TypeTy, IdentTy, ExprTy>;
using Severity = tomp::clause::SeverityT<TypeTy, IdentTy, ExprTy>;
using Shared = tomp::clause::SharedT<TypeTy, IdentTy, ExprTy>;
using Simdlen = tomp::clause::SimdlenT<TypeTy, IdentTy, ExprTy>;
using Simd = tomp::clause::SimdT<TypeTy, IdentTy, ExprTy>;
using Sizes = tomp::clause::SizesT<TypeTy, IdentTy, ExprTy>;
using TaskReduction = tomp::clause::TaskReductionT<TypeTy, IdentTy, ExprTy>;
using ThreadLimit = tomp::clause::ThreadLimitT<TypeTy, IdentTy, ExprTy>;
using Threadprivate = tomp::clause::ThreadprivateT<TypeTy, IdentTy, ExprTy>;
using Threads = tomp::clause::ThreadsT<TypeTy, IdentTy, ExprTy>;
using To = tomp::clause::ToT<TypeTy, IdentTy, ExprTy>;
using UnifiedAddress = tomp::clause::UnifiedAddressT<TypeTy, IdentTy, ExprTy>;
using UnifiedSharedMemory =
    tomp::clause::UnifiedSharedMemoryT<TypeTy, IdentTy, ExprTy>;
using Uniform = tomp::clause::UniformT<TypeTy, IdentTy, ExprTy>;
using Unknown = tomp::clause::UnknownT<TypeTy, IdentTy, ExprTy>;
using Untied = tomp::clause::UntiedT<TypeTy, IdentTy, ExprTy>;
using Update = tomp::clause::UpdateT<TypeTy, IdentTy, ExprTy>;
using UseDeviceAddr = tomp::clause::UseDeviceAddrT<TypeTy, IdentTy, ExprTy>;
using UseDevicePtr = tomp::clause::UseDevicePtrT<TypeTy, IdentTy, ExprTy>;
using UsesAllocators = tomp::clause::UsesAllocatorsT<TypeTy, IdentTy, ExprTy>;
using Use = tomp::clause::UseT<TypeTy, IdentTy, ExprTy>;
using Weak = tomp::clause::WeakT<TypeTy, IdentTy, ExprTy>;
using When = tomp::clause::WhenT<TypeTy, IdentTy, ExprTy>;
using Write = tomp::clause::WriteT<TypeTy, IdentTy, ExprTy>;
} // namespace clause

struct Clause : public tomp::ClauseT<TypeTy, IdentTy, ExprTy> {
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
