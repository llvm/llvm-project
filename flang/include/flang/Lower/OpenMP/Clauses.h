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
#include "flang/Evaluate/type.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/OpenMP/ClauseT.h"

#include <optional>
#include <type_traits>
#include <utility>

namespace Fortran::semantics {
class Symbol;
}

namespace Fortran::lower::omp {
using namespace Fortran;
using SomeExpr = semantics::SomeExpr;
using MaybeExpr = semantics::MaybeExpr;
using TypeTy = evaluate::DynamicType;

template <typename ExprTy>
struct IdTyTemplate {
  // "symbol" is always non-null for id's of actual objects.
  Fortran::semantics::Symbol *symbol;
  std::optional<ExprTy> designator;

  bool operator==(const IdTyTemplate &other) const {
    // If symbols are different, then the objects are different.
    if (symbol != other.symbol)
      return false;
    if (symbol == nullptr)
      return true;
    // Equal symbols don't necessarily indicate identical objects,
    // for example, a derived object component may use a single symbol,
    // which will refer to different objects for different designators,
    // e.g. a%c and b%c.
    return designator == other.designator;
  }

  // Defining an "ordering" which allows types derived from this to be
  // utilised in maps and other containers that require comparison
  // operators for ordering
  bool operator<(const IdTyTemplate &other) const {
    return symbol < other.symbol;
  }

  operator bool() const { return symbol != nullptr; }
};

using ExprTy = SomeExpr;

template <typename T>
using List = tomp::ListT<T>;
} // namespace Fortran::lower::omp

// Specialization of the ObjectT template
namespace tomp::type {
template <>
struct ObjectT<Fortran::lower::omp::IdTyTemplate<Fortran::lower::omp::ExprTy>,
               Fortran::lower::omp::ExprTy> {
  using IdTy = Fortran::lower::omp::IdTyTemplate<Fortran::lower::omp::ExprTy>;
  using ExprTy = Fortran::lower::omp::ExprTy;

  IdTy id() const { return identity; }
  Fortran::semantics::Symbol *sym() const { return identity.symbol; }
  const std::optional<ExprTy> &ref() const { return identity.designator; }

  bool operator<(const ObjectT<IdTy, ExprTy> &other) const {
    return identity < other.identity;
  }

  IdTy identity;
};
} // namespace tomp::type

namespace Fortran::lower::omp {
using IdTy = IdTyTemplate<ExprTy>;
}

namespace std {
template <>
struct hash<Fortran::lower::omp::IdTy> {
  size_t operator()(const Fortran::lower::omp::IdTy &id) const {
    return static_cast<size_t>(reinterpret_cast<uintptr_t>(id.symbol));
  }
};
} // namespace std

namespace Fortran::lower::omp {
using Object = tomp::ObjectT<IdTy, ExprTy>;
using ObjectList = tomp::ObjectListT<IdTy, ExprTy>;

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

inline ObjectList makeObjects(const parser::OmpObjectList &objects,
                              semantics::SemanticsContext &semaCtx) {
  return makeList(objects.v, makeObjectFn(semaCtx));
}

template <typename FuncTy, //
          typename ArgTy,  //
          typename ResultTy = std::invoke_result_t<FuncTy, ArgTy>>
std::optional<ResultTy> maybeApply(FuncTy &&func,
                                   const std::optional<ArgTy> &arg) {
  if (!arg)
    return std::nullopt;
  return func(*arg);
}

template <           //
    typename FuncTy, //
    typename ArgTy,  //
    typename ResultTy =
        std::invoke_result_t<FuncTy, decltype(std::declval<ArgTy>().v)>>
std::optional<ResultTy> maybeApplyToV(FuncTy &&func, const ArgTy *arg) {
  if (!arg)
    return std::nullopt;
  return func(arg->v);
}

std::optional<Object> getBaseObject(const Object &object,
                                    semantics::SemanticsContext &semaCtx);

namespace clause {
using Range = tomp::type::RangeT<ExprTy>;
using Mapper = tomp::type::MapperT<IdTy, ExprTy>;
using Iterator = tomp::type::IteratorT<TypeTy, IdTy, ExprTy>;
using IteratorSpecifier = tomp::type::IteratorSpecifierT<TypeTy, IdTy, ExprTy>;
using DefinedOperator = tomp::type::DefinedOperatorT<IdTy, ExprTy>;
using ProcedureDesignator = tomp::type::ProcedureDesignatorT<IdTy, ExprTy>;
using ReductionOperator = tomp::type::ReductionIdentifierT<IdTy, ExprTy>;
using DependenceType = tomp::type::DependenceType;
using Prescriptiveness = tomp::type::Prescriptiveness;

// "Requires" clauses are handled early on, and the aggregated information
// is stored in the Symbol details of modules, programs, and subprograms.
// These clauses are still handled here to cover all alternatives in the
// main clause variant.

using Absent = tomp::clause::AbsentT<TypeTy, IdTy, ExprTy>;
using AcqRel = tomp::clause::AcqRelT<TypeTy, IdTy, ExprTy>;
using Acquire = tomp::clause::AcquireT<TypeTy, IdTy, ExprTy>;
using AdjustArgs = tomp::clause::AdjustArgsT<TypeTy, IdTy, ExprTy>;
using Affinity = tomp::clause::AffinityT<TypeTy, IdTy, ExprTy>;
using Aligned = tomp::clause::AlignedT<TypeTy, IdTy, ExprTy>;
using Align = tomp::clause::AlignT<TypeTy, IdTy, ExprTy>;
using Allocate = tomp::clause::AllocateT<TypeTy, IdTy, ExprTy>;
using Allocator = tomp::clause::AllocatorT<TypeTy, IdTy, ExprTy>;
using AppendArgs = tomp::clause::AppendArgsT<TypeTy, IdTy, ExprTy>;
using AtomicDefaultMemOrder =
    tomp::clause::AtomicDefaultMemOrderT<TypeTy, IdTy, ExprTy>;
using At = tomp::clause::AtT<TypeTy, IdTy, ExprTy>;
using Bind = tomp::clause::BindT<TypeTy, IdTy, ExprTy>;
using Capture = tomp::clause::CaptureT<TypeTy, IdTy, ExprTy>;
using Collapse = tomp::clause::CollapseT<TypeTy, IdTy, ExprTy>;
using Compare = tomp::clause::CompareT<TypeTy, IdTy, ExprTy>;
using Contains = tomp::clause::ContainsT<TypeTy, IdTy, ExprTy>;
using Copyin = tomp::clause::CopyinT<TypeTy, IdTy, ExprTy>;
using Copyprivate = tomp::clause::CopyprivateT<TypeTy, IdTy, ExprTy>;
using Defaultmap = tomp::clause::DefaultmapT<TypeTy, IdTy, ExprTy>;
using Default = tomp::clause::DefaultT<TypeTy, IdTy, ExprTy>;
using Depend = tomp::clause::DependT<TypeTy, IdTy, ExprTy>;
using Destroy = tomp::clause::DestroyT<TypeTy, IdTy, ExprTy>;
using Detach = tomp::clause::DetachT<TypeTy, IdTy, ExprTy>;
using Device = tomp::clause::DeviceT<TypeTy, IdTy, ExprTy>;
using DeviceType = tomp::clause::DeviceTypeT<TypeTy, IdTy, ExprTy>;
using DistSchedule = tomp::clause::DistScheduleT<TypeTy, IdTy, ExprTy>;
using Doacross = tomp::clause::DoacrossT<TypeTy, IdTy, ExprTy>;
using DynamicAllocators =
    tomp::clause::DynamicAllocatorsT<TypeTy, IdTy, ExprTy>;
using Enter = tomp::clause::EnterT<TypeTy, IdTy, ExprTy>;
using Exclusive = tomp::clause::ExclusiveT<TypeTy, IdTy, ExprTy>;
using Fail = tomp::clause::FailT<TypeTy, IdTy, ExprTy>;
using Filter = tomp::clause::FilterT<TypeTy, IdTy, ExprTy>;
using Final = tomp::clause::FinalT<TypeTy, IdTy, ExprTy>;
using Firstprivate = tomp::clause::FirstprivateT<TypeTy, IdTy, ExprTy>;
using From = tomp::clause::FromT<TypeTy, IdTy, ExprTy>;
using Full = tomp::clause::FullT<TypeTy, IdTy, ExprTy>;
using Grainsize = tomp::clause::GrainsizeT<TypeTy, IdTy, ExprTy>;
using HasDeviceAddr = tomp::clause::HasDeviceAddrT<TypeTy, IdTy, ExprTy>;
using Hint = tomp::clause::HintT<TypeTy, IdTy, ExprTy>;
using Holds = tomp::clause::HoldsT<TypeTy, IdTy, ExprTy>;
using If = tomp::clause::IfT<TypeTy, IdTy, ExprTy>;
using Inbranch = tomp::clause::InbranchT<TypeTy, IdTy, ExprTy>;
using Inclusive = tomp::clause::InclusiveT<TypeTy, IdTy, ExprTy>;
using Indirect = tomp::clause::IndirectT<TypeTy, IdTy, ExprTy>;
using Init = tomp::clause::InitT<TypeTy, IdTy, ExprTy>;
using InReduction = tomp::clause::InReductionT<TypeTy, IdTy, ExprTy>;
using IsDevicePtr = tomp::clause::IsDevicePtrT<TypeTy, IdTy, ExprTy>;
using Lastprivate = tomp::clause::LastprivateT<TypeTy, IdTy, ExprTy>;
using Linear = tomp::clause::LinearT<TypeTy, IdTy, ExprTy>;
using Link = tomp::clause::LinkT<TypeTy, IdTy, ExprTy>;
using Map = tomp::clause::MapT<TypeTy, IdTy, ExprTy>;
using Match = tomp::clause::MatchT<TypeTy, IdTy, ExprTy>;
using Mergeable = tomp::clause::MergeableT<TypeTy, IdTy, ExprTy>;
using Message = tomp::clause::MessageT<TypeTy, IdTy, ExprTy>;
using NoOpenmp = tomp::clause::NoOpenmpT<TypeTy, IdTy, ExprTy>;
using NoOpenmpRoutines = tomp::clause::NoOpenmpRoutinesT<TypeTy, IdTy, ExprTy>;
using NoParallelism = tomp::clause::NoParallelismT<TypeTy, IdTy, ExprTy>;
using Nocontext = tomp::clause::NocontextT<TypeTy, IdTy, ExprTy>;
using Nogroup = tomp::clause::NogroupT<TypeTy, IdTy, ExprTy>;
using Nontemporal = tomp::clause::NontemporalT<TypeTy, IdTy, ExprTy>;
using Notinbranch = tomp::clause::NotinbranchT<TypeTy, IdTy, ExprTy>;
using Novariants = tomp::clause::NovariantsT<TypeTy, IdTy, ExprTy>;
using Nowait = tomp::clause::NowaitT<TypeTy, IdTy, ExprTy>;
using NumTasks = tomp::clause::NumTasksT<TypeTy, IdTy, ExprTy>;
using NumTeams = tomp::clause::NumTeamsT<TypeTy, IdTy, ExprTy>;
using NumThreads = tomp::clause::NumThreadsT<TypeTy, IdTy, ExprTy>;
using OmpxAttribute = tomp::clause::OmpxAttributeT<TypeTy, IdTy, ExprTy>;
using OmpxBare = tomp::clause::OmpxBareT<TypeTy, IdTy, ExprTy>;
using OmpxDynCgroupMem = tomp::clause::OmpxDynCgroupMemT<TypeTy, IdTy, ExprTy>;
using Ordered = tomp::clause::OrderedT<TypeTy, IdTy, ExprTy>;
using Order = tomp::clause::OrderT<TypeTy, IdTy, ExprTy>;
using Otherwise = tomp::clause::OtherwiseT<TypeTy, IdTy, ExprTy>;
using Partial = tomp::clause::PartialT<TypeTy, IdTy, ExprTy>;
using Priority = tomp::clause::PriorityT<TypeTy, IdTy, ExprTy>;
using Private = tomp::clause::PrivateT<TypeTy, IdTy, ExprTy>;
using ProcBind = tomp::clause::ProcBindT<TypeTy, IdTy, ExprTy>;
using Read = tomp::clause::ReadT<TypeTy, IdTy, ExprTy>;
using Reduction = tomp::clause::ReductionT<TypeTy, IdTy, ExprTy>;
using Relaxed = tomp::clause::RelaxedT<TypeTy, IdTy, ExprTy>;
using Release = tomp::clause::ReleaseT<TypeTy, IdTy, ExprTy>;
using ReverseOffload = tomp::clause::ReverseOffloadT<TypeTy, IdTy, ExprTy>;
using Safelen = tomp::clause::SafelenT<TypeTy, IdTy, ExprTy>;
using Schedule = tomp::clause::ScheduleT<TypeTy, IdTy, ExprTy>;
using SeqCst = tomp::clause::SeqCstT<TypeTy, IdTy, ExprTy>;
using Severity = tomp::clause::SeverityT<TypeTy, IdTy, ExprTy>;
using Shared = tomp::clause::SharedT<TypeTy, IdTy, ExprTy>;
using Simdlen = tomp::clause::SimdlenT<TypeTy, IdTy, ExprTy>;
using Simd = tomp::clause::SimdT<TypeTy, IdTy, ExprTy>;
using Sizes = tomp::clause::SizesT<TypeTy, IdTy, ExprTy>;
using Permutation = tomp::clause::PermutationT<TypeTy, IdTy, ExprTy>;
using TaskReduction = tomp::clause::TaskReductionT<TypeTy, IdTy, ExprTy>;
using ThreadLimit = tomp::clause::ThreadLimitT<TypeTy, IdTy, ExprTy>;
using Threads = tomp::clause::ThreadsT<TypeTy, IdTy, ExprTy>;
using To = tomp::clause::ToT<TypeTy, IdTy, ExprTy>;
using UnifiedAddress = tomp::clause::UnifiedAddressT<TypeTy, IdTy, ExprTy>;
using UnifiedSharedMemory =
    tomp::clause::UnifiedSharedMemoryT<TypeTy, IdTy, ExprTy>;
using Uniform = tomp::clause::UniformT<TypeTy, IdTy, ExprTy>;
using Unknown = tomp::clause::UnknownT<TypeTy, IdTy, ExprTy>;
using Untied = tomp::clause::UntiedT<TypeTy, IdTy, ExprTy>;
using Update = tomp::clause::UpdateT<TypeTy, IdTy, ExprTy>;
using UseDeviceAddr = tomp::clause::UseDeviceAddrT<TypeTy, IdTy, ExprTy>;
using UseDevicePtr = tomp::clause::UseDevicePtrT<TypeTy, IdTy, ExprTy>;
using UsesAllocators = tomp::clause::UsesAllocatorsT<TypeTy, IdTy, ExprTy>;
using Use = tomp::clause::UseT<TypeTy, IdTy, ExprTy>;
using Weak = tomp::clause::WeakT<TypeTy, IdTy, ExprTy>;
using When = tomp::clause::WhenT<TypeTy, IdTy, ExprTy>;
using Write = tomp::clause::WriteT<TypeTy, IdTy, ExprTy>;
} // namespace clause

using tomp::type::operator==;

struct CancellationConstructType {
  using EmptyTrait = std::true_type;
};
struct Depobj {
  using EmptyTrait = std::true_type;
};
struct Flush {
  using EmptyTrait = std::true_type;
};
struct MemoryOrder {
  using EmptyTrait = std::true_type;
};
struct Threadprivate {
  using EmptyTrait = std::true_type;
};

using ClauseBase = tomp::ClauseT<TypeTy, IdTy, ExprTy,
                                 // Extras...
                                 CancellationConstructType, Depobj, Flush,
                                 MemoryOrder, Threadprivate>;

struct Clause : public ClauseBase {
  Clause(ClauseBase &&base, const parser::CharBlock source = {})
      : ClauseBase(std::move(base)), source(source) {}
  // "source" will be ignored by tomp::type::operator==.
  parser::CharBlock source;
};

template <typename Specific>
Clause makeClause(llvm::omp::Clause id, Specific &&specific,
                  parser::CharBlock source = {}) {
  return Clause(typename Clause::BaseT{id, specific}, source);
}

Clause makeClause(const parser::OmpClause &cls,
                  semantics::SemanticsContext &semaCtx);

List<Clause> makeClauses(const parser::OmpClauseList &clauses,
                         semantics::SemanticsContext &semaCtx);

bool transferLocations(const List<Clause> &from, List<Clause> &to);
} // namespace Fortran::lower::omp

#endif // FORTRAN_LOWER_OPENMP_CLAUSES_H
