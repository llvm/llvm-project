//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef OPENMPCLAUSES_H
#define OPENMPCLAUSES_H

#include "flang/Common/enum-class.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/Frontend/OpenMP/OMP.h.inc"

static llvm::ArrayRef<llvm::omp::Directive> getWorksharing() {
  static llvm::omp::Directive worksharing[] = {
      llvm::omp::Directive::OMPD_do,     llvm::omp::Directive::OMPD_for,
      llvm::omp::Directive::OMPD_scope,  llvm::omp::Directive::OMPD_sections,
      llvm::omp::Directive::OMPD_single, llvm::omp::Directive::OMPD_workshare,
  };
  return worksharing;
}

static llvm::ArrayRef<llvm::omp::Directive> getWorksharingLoop() {
  static llvm::omp::Directive worksharingLoop[] = {
      llvm::omp::Directive::OMPD_do,
      llvm::omp::Directive::OMPD_for,
  };
  return worksharingLoop;
}

namespace detail {
template <typename Container, typename Predicate>
typename std::remove_reference_t<Container>::iterator
find_unique(Container &&container, Predicate &&pred) {
  auto first = std::find_if(container.begin(), container.end(), pred);
  if (first == container.end())
    return first;
  auto second = std::find_if(std::next(first), container.end(), pred);
  if (second == container.end())
    return first;
  return container.end();
}
} // namespace detail

namespace tomp {

template <typename T>
using ListT = std::vector<T>;

// A specialization of ObjectT<Id, Expr> must provide the following definitions:
// {
//    using IdType = Id;
//    using ExprType = Expr;
//
//    auto id() const -> Id {
//      return the identifier of the object (for use in tests for
//         presence/absence of the object)
//    }
//
//    auto ref() const -> const Expr& {
//      return the expression accessing (referencing) the object
//    }
// }
//
// For example, the ObjectT instance created for "var[x+1]" would have
// the `id()` return the identifier for `var`, and the `ref()` return the
// representation of the array-access `var[x+1]`.
template <typename Id, typename Expr>
struct ObjectT;

template <typename I, typename E>
using ObjectListT = ListT<ObjectT<I, E>>;

namespace clause {
// Helper objects

template <typename I, typename E>
struct DefinedOperatorT {
  struct DefinedOpName {
    using WrapperTrait = std::true_type;
    ObjectT<I, E> v;
  };
  ENUM_CLASS(IntrinsicOperator, Power, Multiply, Divide, Add, Subtract, Concat,
             LT, LE, EQ, NE, GE, GT, NOT, AND, OR, EQV, NEQV)
  using UnionTrait = std::true_type;
  std::variant<DefinedOpName, IntrinsicOperator> u;
};

template <typename I, typename E>
struct ProcedureDesignatorT {
  using WrapperTrait = std::true_type;
  ObjectT<I, E> v;
};

template <typename I, typename E>
struct ReductionOperatorT {
  using UnionTrait = std::true_type;
  std::variant<DefinedOperatorT<I, E>, ProcedureDesignatorT<I, E>> u;
};

template <typename I, typename E>
struct AcqRelT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct AcquireT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct AdjustArgsT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct AffinityT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct AlignT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct AppendArgsT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct AtT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct BindT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct CancellationConstructTypeT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct CaptureT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct CompareT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct DepobjT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct DestroyT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct DetachT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct DoacrossT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct DynamicAllocatorsT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct ExclusiveT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct FailT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct FlushT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct FullT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct InbranchT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct InclusiveT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct IndirectT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct InitT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct MatchT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct MemoryOrderT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct MergeableT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct MessageT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct NogroupT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct NotinbranchT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct NowaitT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct OmpxAttributeT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct OmpxBareT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct ReadT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct RelaxedT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct ReleaseT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct ReverseOffloadT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct SeqCstT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct SeverityT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct SimdT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct ThreadprivateT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct ThreadsT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct UnifiedAddressT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct UnifiedSharedMemoryT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct UnknownT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct UntiedT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct UpdateT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct UseT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct UsesAllocatorsT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct WeakT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct WhenT {
  using EmptyTrait = std::true_type;
};
template <typename I, typename E>
struct WriteT {
  using EmptyTrait = std::true_type;
};

template <typename I, typename E>
struct AlignedT {
  using TupleTrait = std::true_type;
  std::tuple<ObjectListT<I, E>, std::optional<E>> t;
};

template <typename I, typename E>
struct AllocateT {
  struct Modifier {
    struct Allocator {
      using WrapperTrait = std::true_type;
      E v;
    };
    struct Align {
      using WrapperTrait = std::true_type;
      E v;
    };
    struct ComplexModifier {
      using TupleTrait = std::true_type;
      std::tuple<Allocator, Align> t;
    };
    using UnionTrait = std::true_type;
    std::variant<Allocator, ComplexModifier, Align> u;
  };
  using TupleTrait = std::true_type;
  std::tuple<std::optional<Modifier>, ObjectListT<I, E>> t;
};

template <typename I, typename E>
struct AllocatorT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct AtomicDefaultMemOrderT {
  using WrapperTrait = std::true_type;
  // XXX common::OmpAtomicDefaultMemOrderType v;
  ENUM_CLASS(OmpAtomicDefaultMemOrderType, SeqCst, AcqRel, Relaxed)
  OmpAtomicDefaultMemOrderType v;
};

template <typename I, typename E>
struct CollapseT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct CopyinT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct CopyprivateT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct DefaultmapT {
  ENUM_CLASS(ImplicitBehavior, Alloc, To, From, Tofrom, Firstprivate, None,
             Default)
  ENUM_CLASS(VariableCategory, Scalar, Aggregate, Allocatable, Pointer)
  using TupleTrait = std::true_type;
  std::tuple<ImplicitBehavior, std::optional<VariableCategory>> t;
};

template <typename I, typename E>
struct DefaultT {
  ENUM_CLASS(Type, Private, Firstprivate, Shared, None)
  using WrapperTrait = std::true_type;
  Type v;
};

template <typename I, typename E>
struct DependT {
  struct Source {
    using EmptyTrait = std::true_type;
  };
  struct Sink {
    using Length = std::tuple<DefinedOperatorT<I, E>, E>;
    using Vec = std::tuple<ObjectT<I, E>, std::optional<Length>>;
    using WrapperTrait = std::true_type;
    ListT<Vec> v;
  };
  ENUM_CLASS(Type, In, Out, Inout, Source, Sink)
  struct InOut {
    using TupleTrait = std::true_type;
    std::tuple<Type, ObjectListT<I, E>> t;
  };
  using UnionTrait = std::true_type;
  std::variant<Source, Sink, InOut> u;
};

template <typename I, typename E>
struct DeviceT {
  ENUM_CLASS(DeviceModifier, Ancestor, Device_Num)
  using TupleTrait = std::true_type;
  std::tuple<std::optional<DeviceModifier>, E> t;
};

template <typename I, typename E>
struct DeviceTypeT {
  ENUM_CLASS(Type, Any, Host, Nohost)
  using WrapperTrait = std::true_type;
  Type v;
};

template <typename I, typename E>
struct DistScheduleT {
  using WrapperTrait = std::true_type;
  std::optional<E> v;
};

template <typename I, typename E>
struct EnterT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct FilterT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct FinalT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct FirstprivateT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct FromT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct GrainsizeT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct HasDeviceAddrT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct HintT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct IfT {
  ENUM_CLASS(DirectiveNameModifier, Parallel, Simd, Target, TargetData,
             TargetEnterData, TargetExitData, TargetUpdate, Task, Taskloop,
             Teams)
  using TupleTrait = std::true_type;
  std::tuple<std::optional<DirectiveNameModifier>, E> t;
};

template <typename I, typename E>
struct InReductionT {
  using TupleTrait = std::true_type;
  std::tuple<ReductionOperatorT<I, E>, ObjectListT<I, E>> t;
};

template <typename I, typename E>
struct IsDevicePtrT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct LastprivateT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct LinearT {
  struct Modifier {
    ENUM_CLASS(Type, Exp, Val, Uval)
    using WrapperTrait = std::true_type;
    Type v;
  };
  using TupleTrait = std::true_type;
  std::tuple<std::optional<Modifier>, ObjectListT<I, E>, std::optional<E>> t;
};

template <typename I, typename E>
struct LinkT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct MapT {
  struct MapType {
    struct Always {
      using EmptyTrait = std::true_type;
    };
    ENUM_CLASS(Type, To, From, Tofrom, Alloc, Release, Delete)
    using TupleTrait = std::true_type;
    std::tuple<std::optional<Always>, Type> t;
  };
  using TupleTrait = std::true_type;
  std::tuple<std::optional<MapType>, ObjectListT<I, E>> t;
};

template <typename I, typename E>
struct NocontextT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct NontemporalT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct NovariantsT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct NumTasksT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct NumTeamsT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct NumThreadsT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct OmpxDynCgroupMemT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct OrderedT {
  using WrapperTrait = std::true_type;
  std::optional<E> v;
};

template <typename I, typename E>
struct OrderT {
  ENUM_CLASS(Kind, Reproducible, Unconstrained)
  ENUM_CLASS(Type, Concurrent)
  using TupleTrait = std::true_type;
  std::tuple<std::optional<Kind>, Type> t;
};

template <typename I, typename E>
struct PartialT {
  using WrapperTrait = std::true_type;
  std::optional<E> v;
};

template <typename I, typename E>
struct PriorityT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct PrivateT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct ProcBindT {
  ENUM_CLASS(Type, Close, Master, Spread, Primary)
  using WrapperTrait = std::true_type;
  Type v;
};

template <typename I, typename E>
struct ReductionT {
  using TupleTrait = std::true_type;
  std::tuple<ReductionOperatorT<I, E>, ObjectListT<I, E>> t;
};

template <typename I, typename E>
struct SafelenT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct ScheduleT {
  ENUM_CLASS(ModType, Monotonic, Nonmonotonic, Simd)
  struct ScheduleModifier {
    using TupleTrait = std::true_type;
    std::tuple<ModType, std::optional<ModType>> t;
  };
  ENUM_CLASS(ScheduleType, Static, Dynamic, Guided, Auto, Runtime)
  using TupleTrait = std::true_type;
  std::tuple<std::optional<ScheduleModifier>, ScheduleType, std::optional<E>> t;
};

template <typename I, typename E>
struct SharedT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct SimdlenT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct SizesT {
  using WrapperTrait = std::true_type;
  ListT<E> v;
};

template <typename I, typename E>
struct TaskReductionT {
  using TupleTrait = std::true_type;
  std::tuple<ReductionOperatorT<I, E>, ObjectListT<I, E>> t;
};

template <typename I, typename E>
struct ThreadLimitT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename I, typename E>
struct ToT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct UniformT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct UseDeviceAddrT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
struct UseDevicePtrT {
  using WrapperTrait = std::true_type;
  ObjectListT<I, E> v;
};

template <typename I, typename E>
using UnionOfAllClausesT = std::variant<
    AcqRelT<I, E>, AcquireT<I, E>, AdjustArgsT<I, E>, AffinityT<I, E>,
    AlignT<I, E>, AlignedT<I, E>, AllocateT<I, E>, AllocatorT<I, E>,
    AppendArgsT<I, E>, AtT<I, E>, AtomicDefaultMemOrderT<I, E>, BindT<I, E>,
    CancellationConstructTypeT<I, E>, CaptureT<I, E>, CollapseT<I, E>,
    CompareT<I, E>, CopyprivateT<I, E>, CopyinT<I, E>, DefaultT<I, E>,
    DefaultmapT<I, E>, DependT<I, E>, DepobjT<I, E>, DestroyT<I, E>,
    DetachT<I, E>, DeviceT<I, E>, DeviceTypeT<I, E>, DistScheduleT<I, E>,
    DoacrossT<I, E>, DynamicAllocatorsT<I, E>, EnterT<I, E>, ExclusiveT<I, E>,
    FailT<I, E>, FilterT<I, E>, FinalT<I, E>, FirstprivateT<I, E>, FlushT<I, E>,
    FromT<I, E>, FullT<I, E>, GrainsizeT<I, E>, HasDeviceAddrT<I, E>,
    HintT<I, E>, IfT<I, E>, InReductionT<I, E>, InbranchT<I, E>,
    InclusiveT<I, E>, IndirectT<I, E>, InitT<I, E>, IsDevicePtrT<I, E>,
    LastprivateT<I, E>, LinearT<I, E>, LinkT<I, E>, MapT<I, E>, MatchT<I, E>,
    MemoryOrderT<I, E>, MergeableT<I, E>, MessageT<I, E>, NogroupT<I, E>,
    NowaitT<I, E>, NocontextT<I, E>, NontemporalT<I, E>, NotinbranchT<I, E>,
    NovariantsT<I, E>, NumTasksT<I, E>, NumTeamsT<I, E>, NumThreadsT<I, E>,
    OmpxAttributeT<I, E>, OmpxDynCgroupMemT<I, E>, OmpxBareT<I, E>,
    OrderT<I, E>, OrderedT<I, E>, PartialT<I, E>, PriorityT<I, E>,
    PrivateT<I, E>, ProcBindT<I, E>, ReadT<I, E>, ReductionT<I, E>,
    RelaxedT<I, E>, ReleaseT<I, E>, ReverseOffloadT<I, E>, SafelenT<I, E>,
    ScheduleT<I, E>, SeqCstT<I, E>, SeverityT<I, E>, SharedT<I, E>, SimdT<I, E>,
    SimdlenT<I, E>, SizesT<I, E>, TaskReductionT<I, E>, ThreadLimitT<I, E>,
    ThreadprivateT<I, E>, ThreadsT<I, E>, ToT<I, E>, UnifiedAddressT<I, E>,
    UnifiedSharedMemoryT<I, E>, UniformT<I, E>, UnknownT<I, E>, UntiedT<I, E>,
    UpdateT<I, E>, UseT<I, E>, UseDeviceAddrT<I, E>, UseDevicePtrT<I, E>,
    UsesAllocatorsT<I, E>, WeakT<I, E>, WhenT<I, E>, WriteT<I, E>>;
} // namespace clause

template <typename Id, typename Expr>
struct ClauseT {
  llvm::omp::Clause id; // The numeric id of the clause
  using UnionTrait = std::true_type;
  clause::UnionOfAllClausesT<Id, Expr> u;
};

// --------------------------------------------------------------------
template <typename IdType, typename ExprType>
struct DirectiveInfo {
  llvm::omp::Directive id = llvm::omp::Directive::OMPD_unknown;
  llvm::SmallVector<const tomp::ClauseT<IdType, ExprType> *> clauses;
};

template <typename I, typename E>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const DirectiveInfo<I, E> &di) {
  os << llvm::omp::getOpenMPDirectiveName(di.id) << " -> {";
  for (int i = 0, e = di.clauses.size(); i != e; ++i) {
    if (i != 0)
      os << ", ";
    os << llvm::omp::getOpenMPClauseName(di.clauses[i]->id);
  }
  os << '}';
  return os;
}

template <typename IdTy, typename ExprTy, typename HelperTy>
struct CompositeInfoBase {
  using IdType = IdTy;
  using ExprType = ExprTy;
  using HelperType = HelperTy;
  using ObjectType = tomp::ObjectT<IdType, ExprType>;
  using ClauseType = tomp::ClauseT<IdType, ExprType>;

  using ClauseSet = llvm::DenseSet<const ClauseType *>;

  CompositeInfoBase(uint32_t ver, llvm::omp::Directive dir, HelperType &hlp)
      : version(ver), construct(dir), helper(hlp) {}

  void add(const ClauseType *node) { nodes.push_back(node); }

  bool split();

  DirectiveInfo<IdType, ExprType> *findDirective(llvm::omp::Directive dirId) {
    for (DirectiveInfo<IdType, ExprType> &dir : leafs) {
      if (dir.id == dirId)
        return &dir;
    }
    return nullptr;
  }
  ClauseSet *findClauses(const ObjectType &object) {
    if (auto found = syms.find(object.id()); found != syms.end())
      return &found->second;
    return nullptr;
  }

  uint32_t version;
  llvm::omp::Directive construct;

  // Leafs are ordered outer to inner.
  llvm::SmallVector<DirectiveInfo<IdType, ExprType>> leafs;

private:
  template <typename S>
  ClauseType *makeClause(llvm::omp::Clause clauseId, S &&specific) {
    auto clause = ClauseType{clauseId, std::move(specific)};
    auto *addr = helper.makeClause(std::move(clause));
    return static_cast<ClauseType *>(addr);
  }

  void addClauseSymsToMap(const ObjectType &object, const ClauseType *);
  void addClauseSymsToMap(const tomp::ObjectListT<IdType, ExprType> &objects,
                          const ClauseType *);
  void addClauseSymsToMap(const ExprType &item, const ClauseType *);
  void addClauseSymsToMap(const tomp::clause::MapT<IdType, ExprType> &item,
                          const ClauseType *);

  template <typename T>
  void addClauseSymsToMap(const std::optional<T> &item, const ClauseType *);
  template <typename T>
  void addClauseSymsToMap(const tomp::ListT<T> &item, const ClauseType *);
  template <typename... T, size_t... Is>
  void addClauseSymsToMap(const std::tuple<T...> &item, const ClauseType *,
                          std::index_sequence<Is...> = {});
  template <typename T>
  std::enable_if_t<std::is_enum_v<llvm::remove_cvref_t<T>>, void>
  addClauseSymsToMap(T &&item, const ClauseType *);

  template <typename T>
  std::enable_if_t<llvm::remove_cvref_t<T>::EmptyTrait::value, void>
  addClauseSymsToMap(T &&item, const ClauseType *);

  template <typename T>
  std::enable_if_t<llvm::remove_cvref_t<T>::WrapperTrait::value, void>
  addClauseSymsToMap(T &&item, const ClauseType *);

  template <typename T>
  std::enable_if_t<llvm::remove_cvref_t<T>::TupleTrait::value, void>
  addClauseSymsToMap(T &&item, const ClauseType *);

  template <typename T>
  std::enable_if_t<llvm::remove_cvref_t<T>::UnionTrait::value, void>
  addClauseSymsToMap(T &&item, const ClauseType *);

  // Apply a clause to the only directive that allows it. If there are no
  // directives that allow it, or if there is more that one, do not apply
  // anything and return false, otherwise return true.
  bool applyToUnique(const ClauseType *node);

  // Apply a clause to the first directive in given range that allows it.
  // If such a directive does not exist, return false, otherwise return true.
  template <typename Iterator>
  bool applyToFirst(const ClauseType *node,
                    llvm::iterator_range<Iterator> range);

  // Apply a clause to the innermost directive that allows it. If such a
  // directive does not exist, return false, otherwise return true.
  bool applyToInnermost(const ClauseType *node);

  // Apply a clause to the outermost directive that allows it. If such a
  // directive does not exist, return false, otherwise return true.
  bool applyToOutermost(const ClauseType *node);

  template <typename Predicate>
  bool applyIf(const ClauseType *node, Predicate shouldApply);

  bool applyToAll(const ClauseType *node);

  template <typename Clause>
  bool applyClause(Clause &&clause, const ClauseType *node);

  bool applyClause(const tomp::clause::CollapseT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::PrivateT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::FirstprivateT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::LastprivateT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::SharedT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::DefaultT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::ThreadLimitT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::OrderT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::AllocateT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::ReductionT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::IfT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::LinearT<IdType, ExprType> &clause,
                   const ClauseType *);
  bool applyClause(const tomp::clause::NowaitT<IdType, ExprType> &clause,
                   const ClauseType *);

  HelperType &helper;
  tomp::ListT<const ClauseType *> nodes;

  llvm::DenseMap<IdType, ClauseSet> syms;
  llvm::DenseSet<IdType> mapBases;
};

template <typename I, typename E, typename H>
void CompositeInfoBase<I, E, H>::addClauseSymsToMap(const ObjectType &object,
                                                    const ClauseType *node) {
  syms[object.id()].insert(node);
}

template <typename I, typename E, typename H>
void CompositeInfoBase<I, E, H>::addClauseSymsToMap(
    const tomp::ObjectListT<I, E> &objects, const ClauseType *node) {
  for (auto &object : objects)
    syms[object.id()].insert(node);
}

template <typename I, typename E, typename H>
void CompositeInfoBase<I, E, H>::addClauseSymsToMap(const E &expr,
                                                    const ClauseType *node) {
  // Nothing to do for expressions.
}

template <typename I, typename E, typename H>
void CompositeInfoBase<I, E, H>::addClauseSymsToMap(
    const tomp::clause::MapT<I, E> &item, const ClauseType *node) {
  auto &objects = std::get<tomp::ObjectListT<I, E>>(item.t);
  addClauseSymsToMap(objects, node);
  for (auto &object : objects) {
    if (auto base = helper.getBaseObject(object))
      mapBases.insert(base->id());
  }
}

template <typename I, typename E, typename H>
template <typename T>
void CompositeInfoBase<I, E, H>::addClauseSymsToMap(
    const std::optional<T> &item, const ClauseType *node) {
  if (item)
    addClauseSymsToMap(*item, node);
}

template <typename I, typename E, typename H>
template <typename T>
void CompositeInfoBase<I, E, H>::addClauseSymsToMap(const tomp::ListT<T> &item,
                                                    const ClauseType *node) {
  for (auto &s : item)
    addClauseSymsToMap(s, node);
}

template <typename I, typename E, typename H>
template <typename... T, size_t... Is>
void CompositeInfoBase<I, E, H>::addClauseSymsToMap(
    const std::tuple<T...> &item, const ClauseType *node,
    std::index_sequence<Is...>) {
  (void)node; // Silence strange warning from GCC.
  (addClauseSymsToMap(std::get<Is>(item), node), ...);
}

template <typename I, typename E, typename H>
template <typename T>
std::enable_if_t<std::is_enum_v<llvm::remove_cvref_t<T>>, void>
CompositeInfoBase<I, E, H>::addClauseSymsToMap(T &&item,
                                               const ClauseType *node) {
  // Nothing to do for enums.
}

template <typename I, typename E, typename H>
template <typename T>
std::enable_if_t<llvm::remove_cvref_t<T>::EmptyTrait::value, void>
CompositeInfoBase<I, E, H>::addClauseSymsToMap(T &&item,
                                               const ClauseType *node) {
  // Nothing to do for an empty class.
}

template <typename I, typename E, typename H>
template <typename T>
std::enable_if_t<llvm::remove_cvref_t<T>::WrapperTrait::value, void>
CompositeInfoBase<I, E, H>::addClauseSymsToMap(T &&item,
                                               const ClauseType *node) {
  addClauseSymsToMap(item.v, node);
}

template <typename I, typename E, typename H>
template <typename T>
std::enable_if_t<llvm::remove_cvref_t<T>::TupleTrait::value, void>
CompositeInfoBase<I, E, H>::addClauseSymsToMap(T &&item,
                                               const ClauseType *node) {
  constexpr size_t tuple_size =
      std::tuple_size_v<llvm::remove_cvref_t<decltype(item.t)>>;
  addClauseSymsToMap(item.t, node, std::make_index_sequence<tuple_size>{});
}

template <typename I, typename E, typename H>
template <typename T>
std::enable_if_t<llvm::remove_cvref_t<T>::UnionTrait::value, void>
CompositeInfoBase<I, E, H>::addClauseSymsToMap(T &&item,
                                               const ClauseType *node) {
  std::visit([&](auto &&s) { addClauseSymsToMap(s, node); }, item.u);
}

// Apply a clause to the only directive that allows it. If there are no
// directives that allow it, or if there is more that one, do not apply
// anything and return false, otherwise return true.
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyToUnique(const ClauseType *node) {
  auto unique = detail::find_unique(leafs, [=](const auto &dirInfo) {
    return llvm::omp::isAllowedClauseForDirective(dirInfo.id, node->id,
                                                  version);
  });

  if (unique != leafs.end()) {
    unique->clauses.push_back(node);
    return true;
  }
  return false;
}

// Apply a clause to the first directive in given range that allows it.
// If such a directive does not exist, return false, otherwise return true.
template <typename I, typename E, typename H>
template <typename Iterator>
bool CompositeInfoBase<I, E, H>::applyToFirst(
    const ClauseType *node, llvm::iterator_range<Iterator> range) {
  if (range.empty())
    return false;

  for (DirectiveInfo<I, E> &dir : range) {
    if (!llvm::omp::isAllowedClauseForDirective(dir.id, node->id, version))
      continue;
    dir.clauses.push_back(node);
    return true;
  }
  return false;
}

// Apply a clause to the innermost directive that allows it. If such a
// directive does not exist, return false, otherwise return true.
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyToInnermost(const ClauseType *node) {
  return applyToFirst(node, llvm::reverse(leafs));
}

// Apply a clause to the outermost directive that allows it. If such a
// directive does not exist, return false, otherwise return true.
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyToOutermost(const ClauseType *node) {
  return applyToFirst(node, llvm::iterator_range(leafs));
}

template <typename I, typename E, typename H>
template <typename Predicate>
bool CompositeInfoBase<I, E, H>::applyIf(const ClauseType *node,
                                         Predicate shouldApply) {
  bool applied = false;
  for (DirectiveInfo<I, E> &dir : leafs) {
    if (!llvm::omp::isAllowedClauseForDirective(dir.id, node->id, version))
      continue;
    if (!shouldApply(dir))
      continue;
    dir.clauses.push_back(node);
    applied = true;
  }

  return applied;
}

template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyToAll(const ClauseType *node) {
  return applyIf(node, [](auto) { return true; });
}

template <typename I, typename E, typename H>
template <typename Clause>
bool CompositeInfoBase<I, E, H>::applyClause(Clause &&clause,
                                             const ClauseType *node) {
  // The default behavior is to find the unique directive to which the
  // given clause may be applied. If there are no such directives, or
  // if there are multiple ones, flag an error.
  // From "OpenMP Application Programming Interface", Version 5.2:
  // S Some clauses are permitted only on a single leaf construct of the
  // S combined or composite construct, in which case the effect is as if
  // S the clause is applied to that specific construct. (p339, 31-33)
  if (applyToUnique(node))
    return true;

  return false;
}

// COLLAPSE
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::CollapseT<I, E> &clause, const ClauseType *node) {
  // Apply COLLAPSE to the innermost directive. If it's not one that
  // allows it flag an error.
  if (!leafs.empty()) {
    DirectiveInfo<I, E> &last = leafs.back();

    if (llvm::omp::isAllowedClauseForDirective(last.id, node->id, version)) {
      last.clauses.push_back(node);
      return true;
    }
  }

  //llvm::errs() << "Cannot apply COLLAPSE\n";
  return false;
}

// PRIVATE
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::PrivateT<I, E> &clause, const ClauseType *node) {
  if (applyToInnermost(node))
    return true;
  //llvm::errs() << "Cannot apply PRIVATE\n";
  return false;
}

// FIRSTPRIVATE
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::FirstprivateT<I, E> &clause, const ClauseType *node) {
  bool applied = false;

  // S Section 17.2
  // S The effect of the firstprivate clause is as if it is applied to one
  // S or more leaf constructs as follows:

  // S - To the distribute construct if it is among the constituent constructs;
  // S - To the teams construct if it is among the constituent constructs and
  // S   the distribute construct is not;
  auto hasDistribute = findDirective(llvm::omp::OMPD_distribute);
  auto hasTeams = findDirective(llvm::omp::OMPD_teams);
  if (hasDistribute != nullptr) {
    hasDistribute->clauses.push_back(node);
    applied = true;
    // S If the teams construct is among the constituent constructs and the
    // S effect is not as if the firstprivate clause is applied to it by the
    // S above rules, then the effect is as if the shared clause with the
    // S same list item is applied to the teams construct.
    if (hasTeams != nullptr) {
      auto *shared = makeClause(llvm::omp::Clause::OMPC_shared,
                                tomp::clause::SharedT<I, E>{clause.v});
      hasTeams->clauses.push_back(shared);
    }
  } else if (hasTeams != nullptr) {
    hasTeams->clauses.push_back(node);
    applied = true;
  }

  // S - To a worksharing construct that accepts the clause if one is among
  // S   the constituent constructs;
  auto findWorksharing = [&]() {
    auto worksharing = getWorksharing();
    for (DirectiveInfo<I, E> &dir : leafs) {
      auto found = llvm::find(worksharing, dir.id);
      if (found != std::end(worksharing))
        return &dir;
    }
    return static_cast<DirectiveInfo<I, E> *>(nullptr);
  };

  auto hasWorksharing = findWorksharing();
  if (hasWorksharing != nullptr) {
    hasWorksharing->clauses.push_back(node);
    applied = true;
  }

  // S - To the taskloop construct if it is among the constituent constructs;
  auto hasTaskloop = findDirective(llvm::omp::OMPD_taskloop);
  if (hasTaskloop != nullptr) {
    hasTaskloop->clauses.push_back(node);
    applied = true;
  }

  // S - To the parallel construct if it is among the constituent constructs
  // S   and neither a taskloop construct nor a worksharing construct that
  // S   accepts the clause is among them;
  auto hasParallel = findDirective(llvm::omp::OMPD_parallel);
  if (hasParallel != nullptr) {
    if (hasTaskloop == nullptr && hasWorksharing == nullptr) {
      hasParallel->clauses.push_back(node);
      applied = true;
    } else {
      // S If the parallel construct is among the constituent constructs and
      // S the effect is not as if the firstprivate clause is applied to it by
      // S the above rules, then the effect is as if the shared clause with
      // S the same list item is applied to the parallel construct.
      auto *shared = makeClause(llvm::omp::Clause::OMPC_shared,
                                tomp::clause::SharedT<I, E>{clause.v});
      hasParallel->clauses.push_back(shared);
    }
  }

  // S - To the target construct if it is among the constituent constructs
  // S   and the same list item neither appears in a lastprivate clause nor
  // S   is the base variable or base pointer of a list item that appears in
  // S   a map clause.
  auto inLastprivate = [&](const ObjectType &object) {
    if (ClauseSet *set = findClauses(object)) {
      return llvm::find_if(*set, [](const ClauseType *c) {
               return c->id == llvm::omp::Clause::OMPC_lastprivate;
             }) != set->end();
    }
    return false;
  };

  auto hasTarget = findDirective(llvm::omp::OMPD_target);
  if (hasTarget != nullptr) {
    tomp::ObjectListT<I, E> objects;
    llvm::copy_if(
        clause.v, std::back_inserter(objects), [&](const ObjectType &object) {
          return !inLastprivate(object) && !mapBases.contains(object.id());
        });
    if (!objects.empty()) {
      auto *firstp = makeClause(llvm::omp::Clause::OMPC_firstprivate,
                                tomp::clause::FirstprivateT<I, E>{objects});
      hasTarget->clauses.push_back(firstp);
      applied = true;
    }
  }

  return applied;
}

// LASTPRIVATE
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::LastprivateT<I, E> &clause, const ClauseType *node) {
  bool applied = false;

  // S The effect of the lastprivate clause is as if it is applied to all leaf
  // S constructs that permit the clause.
  if (!applyToAll(node)) {
    //llvm::errs() << "Cannot apply LASTPRIVATE\n";
    return false;
  }

  auto inFirstprivate = [&](const ObjectType &object) {
    if (ClauseSet *set = findClauses(object)) {
      return llvm::find_if(*set, [](const ClauseType *c) {
               return c->id == llvm::omp::Clause::OMPC_firstprivate;
             }) != set->end();
    }
    return false;
  };

  // Prepare list of objects that could end up in a SHARED clause.
  tomp::ObjectListT<I, E> sharedObjects;
  llvm::copy_if(
      clause.v, std::back_inserter(sharedObjects),
      [&](const ObjectType &object) { return !inFirstprivate(object); });

  if (!sharedObjects.empty()) {
    // S If the parallel construct is among the constituent constructs and the
    // S list item is not also specified in the firstprivate clause, then the
    // S effect of the lastprivate clause is as if the shared clause with the
    // S same list item is applied to the parallel construct.
    if (auto hasParallel = findDirective(llvm::omp::OMPD_parallel)) {
      auto *shared = makeClause(llvm::omp::Clause::OMPC_shared,
                                tomp::clause::SharedT<I, E>{sharedObjects});
      hasParallel->clauses.push_back(shared);
      applied = true;
    }

    // S If the teams construct is among the constituent constructs and the
    // S list item is not also specified in the firstprivate clause, then the
    // S effect of the lastprivate clause is as if the shared clause with the
    // S same list item is applied to the teams construct.
    if (auto hasTeams = findDirective(llvm::omp::OMPD_teams)) {
      auto *shared = makeClause(llvm::omp::Clause::OMPC_shared,
                                tomp::clause::SharedT<I, E>{sharedObjects});
      hasTeams->clauses.push_back(shared);
      applied = true;
    }
  }

  // S If the target construct is among the constituent constructs and the
  // S list item is not the base variable or base pointer of a list item that
  // S appears in a map clause, the effect of the lastprivate clause is as if
  // S the same list item appears in a map clause with a map-type of tofrom.
  if (auto hasTarget = findDirective(llvm::omp::OMPD_target)) {
    tomp::ObjectListT<I, E> tofrom;
    llvm::copy_if(clause.v, std::back_inserter(tofrom),
                  [&](const ObjectType &object) {
                    return !mapBases.contains(object.id());
                  });

    if (!tofrom.empty()) {
      using MapType = typename tomp::clause::MapT<I, E>::MapType;
      auto mapType = MapType{{std::nullopt, MapType::Type::Tofrom}};
      auto *map =
          makeClause(llvm::omp::Clause::OMPC_map,
                     tomp::clause::MapT<I, E>{{mapType, std::move(tofrom)}});
      hasTarget->clauses.push_back(map);
      applied = true;
    }
  }

  return applied;
}

// SHARED
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::SharedT<I, E> &clause, const ClauseType *node) {
  // Apply SHARED to the all leafs that allow it.
  if (applyToAll(node))
    return true;
  //llvm::errs() << "Cannot apply SHARED\n";
  return false;
}

// DEFAULT
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::DefaultT<I, E> &clause, const ClauseType *node) {
  // Apply DEFAULT to the all leafs that allow it.
  if (applyToAll(node))
    return true;
  //llvm::errs() << "Cannot apply DEFAULT\n";
  return false;
}

// THREAD_LIMIT
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::ThreadLimitT<I, E> &clause, const ClauseType *node) {
  // Apply THREAD_LIMIT to the all leafs that allow it.
  if (applyToAll(node))
    return true;
  //llvm::errs() << "Cannot apply THREAD_LIMIT\n";
  return false;
}

// ORDER
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::OrderT<I, E> &clause, const ClauseType *node) {
  // Apply ORDER to the all leafs that allow it.
  if (applyToAll(node))
    return true;
  //llvm::errs() << "Cannot apply ORDER\n";
  return false;
}

// ALLOCATE
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::AllocateT<I, E> &clause, const ClauseType *node) {
  // This one needs to be applied at the end, once we know which clauses are
  // assigned to which leaf constructs.

  // S The effect of the allocate clause is as if it is applied to all leaf
  // S constructs that permit the clause and to which a data-sharing attribute
  // S clause that may create a private copy of the same list item is applied.

  auto canMakePrivateCopy = [](llvm::omp::Clause id) {
    switch (id) {
    case llvm::omp::Clause::OMPC_firstprivate:
    case llvm::omp::Clause::OMPC_lastprivate:
    case llvm::omp::Clause::OMPC_private:
      return true;
    default:
      return false;
    }
  };

  bool applied = applyIf(node, [&](const DirectiveInfo<I, E> &dir) {
    return llvm::any_of(dir.clauses, [&](const ClauseType *n) {
      return canMakePrivateCopy(n->id);
    });
  });

  return applied;
}

// REDUCTION
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::ReductionT<I, E> &clause, const ClauseType *node) {
  // S The effect of the reduction clause is as if it is applied to all leaf
  // S constructs that permit the clause, except for the following constructs:
  // S - The parallel construct, when combined with the sections, worksharing-
  // S   loop, loop, or taskloop construct; and
  // S - The teams construct, when combined with the loop construct.
  bool applyToParallel = true, applyToTeams = true;

  auto hasParallel = findDirective(llvm::omp::Directive::OMPD_parallel);
  if (hasParallel) {
    auto exclusions = llvm::concat<const llvm::omp::Directive>(
        getWorksharingLoop(), llvm::ArrayRef{
                                  llvm::omp::Directive::OMPD_loop,
                                  llvm::omp::Directive::OMPD_sections,
                                  llvm::omp::Directive::OMPD_taskloop,
                              });
    auto present = [&](llvm::omp::Directive id) {
      return findDirective(id) != nullptr;
    };

    if (llvm::any_of(exclusions, present))
      applyToParallel = false;
  }

  auto hasTeams = findDirective(llvm::omp::Directive::OMPD_teams);
  if (hasTeams) {
    // The only exclusion is OMPD_loop.
    if (findDirective(llvm::omp::Directive::OMPD_loop))
      applyToTeams = false;
  }

  auto &objects = std::get<tomp::ObjectListT<I, E>>(clause.t);

  tomp::ObjectListT<I, E> sharedObjects;
  llvm::transform(objects, std::back_inserter(sharedObjects),
                  [&](const ObjectType &object) {
                    auto maybeBase = helper.getBaseObject(object);
                    return maybeBase ? *maybeBase : object;
                  });

  // S For the parallel and teams constructs above, the effect of the
  // S reduction clause instead is as if each list item or, for any list
  // S item that is an array item, its corresponding base array or base
  // S pointer appears in a shared clause for the construct.
  if (!sharedObjects.empty()) {
    if (hasParallel && !applyToParallel) {
      auto *shared = makeClause(llvm::omp::Clause::OMPC_shared,
                                tomp::clause::SharedT<I, E>{sharedObjects});
      hasParallel->clauses.push_back(shared);
    }
    if (hasTeams && !applyToTeams) {
      auto *shared = makeClause(llvm::omp::Clause::OMPC_shared,
                                tomp::clause::SharedT<I, E>{sharedObjects});
      hasTeams->clauses.push_back(shared);
    }
  }

  // TODO(not implemented in parser yet): Apply the following.
  // S If the task reduction-modifier is specified, the effect is as if
  // S it only modifies the behavior of the reduction clause on the innermost
  // S leaf construct that accepts the modifier (see Section 5.5.8). If the
  // S inscan reduction-modifier is specified, the effect is as if it modifies
  // S the behavior of the reduction clause on all constructs of the combined
  // S construct to which the clause is applied and that accept the modifier.

  bool applied = applyIf(node, [&](DirectiveInfo<I, E> &dir) {
    if (!applyToParallel && &dir == hasParallel)
      return false;
    if (!applyToTeams && &dir == hasTeams)
      return false;
    return true;
  });

  // S If a list item in a reduction clause on a combined target construct
  // S does not have the same base variable or base pointer as a list item
  // S in a map clause on the construct, then the effect is as if the list
  // S item in the reduction clause appears as a list item in a map clause
  // S with a map-type of tofrom.
  auto hasTarget = findDirective(llvm::omp::Directive::OMPD_target);
  if (hasTarget && leafs.size() > 1) {
    tomp::ObjectListT<I, E> tofrom;
    llvm::copy_if(objects, std::back_inserter(tofrom),
                  [&](const ObjectType &object) {
                    if (auto maybeBase = helper.getBaseObject(object))
                      return !mapBases.contains(maybeBase->id());
                    return !mapBases.contains(object.id()); // XXX is this ok?
                  });
    if (!tofrom.empty()) {
      using MapType = typename tomp::clause::MapT<I, E>::MapType;
      auto mapType = MapType{{std::nullopt, MapType::Type::Tofrom}};
      auto *map =
          makeClause(llvm::omp::Clause::OMPC_map,
                     tomp::clause::MapT<I, E>{{mapType, std::move(tofrom)}});

      hasTarget->clauses.push_back(map);
      applied = true;
    }
  }

  return applied;
}

// IF
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::IfT<I, E> &clause, const ClauseType *node) {
  using DirectiveNameModifier =
      typename tomp::clause::IfT<I, E>::DirectiveNameModifier;
  auto &modifier = std::get<std::optional<DirectiveNameModifier>>(clause.t);

  if (modifier) {
    llvm::omp::Directive dirId = llvm::omp::Directive::OMPD_unknown;

    switch (*modifier) {
    case DirectiveNameModifier::Parallel:
      dirId = llvm::omp::Directive::OMPD_parallel;
      break;
    case DirectiveNameModifier::Simd:
      dirId = llvm::omp::Directive::OMPD_simd;
      break;
    case DirectiveNameModifier::Target:
      dirId = llvm::omp::Directive::OMPD_target;
      break;
    case DirectiveNameModifier::Task:
      dirId = llvm::omp::Directive::OMPD_task;
      break;
    case DirectiveNameModifier::Taskloop:
      dirId = llvm::omp::Directive::OMPD_taskloop;
      break;
    case DirectiveNameModifier::Teams:
      dirId = llvm::omp::Directive::OMPD_teams;
      break;

    case DirectiveNameModifier::TargetData:
    case DirectiveNameModifier::TargetEnterData:
    case DirectiveNameModifier::TargetExitData:
    case DirectiveNameModifier::TargetUpdate:
    default:
      //llvm::errs() << "Invalid modifier in IF clause\n";
      return false;
    }

    if (auto *hasDir = findDirective(dirId)) {
      hasDir->clauses.push_back(node);
      return true;
    }
    //llvm::errs() << "Directive from modifier not found\n";
    return false;
  }

  if (applyToAll(node))
    return true;

  //llvm::errs() << "Cannot apply IF\n";
  return false;
}

// LINEAR
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::LinearT<I, E> &clause, const ClauseType *node) {
  // S The effect of the linear clause is as if it is applied to the innermost
  // S leaf construct.
  if (applyToInnermost(node)) {
    //llvm::errs() << "Cannot apply LINEAR\n";
    return false;
  }

  // The rest is about SIMD.
  if (!findDirective(llvm::omp::OMPD_simd))
    return true;

  // S Additionally, if the list item is not the iteration variable of a
  // S simd or worksharing-loop SIMD construct, the effect on the outer leaf
  // S constructs is as if the list item was specified in firstprivate and
  // S lastprivate clauses on the combined or composite construct, [...]
  //
  // S If a list item of the linear clause is the iteration variable of a
  // S simd or worksharing-loop SIMD construct and it is not declared in
  // S the construct, the effect on the outer leaf constructs is as if the
  // S list item was specified in a lastprivate clause on the combined or
  // S composite construct [...]

  // It's not clear how an object can be listed in a clause AND be the
  // iteration variable of a construct in which is it declared. If an
  // object is declared in the construct, then the declaration is located
  // after the clause listing it.

  std::optional<ObjectType> iterVar = helper.getIterVar();
  const auto &objects = std::get<tomp::ObjectListT<I, E>>(clause.t);

  // Lists of objects that will be used to construct FIRSTPRIVATE and
  // LASTPRIVATE clauses.
  tomp::ObjectListT<I, E> first, last;

  for (const ObjectType &object : objects) {
    last.push_back(object);
    if (iterVar && object.id() != iterVar->id())
      first.push_back(object);
  }

  if (!first.empty()) {
    auto *firstp = makeClause(llvm::omp::Clause::OMPC_firstprivate,
                              tomp::clause::FirstprivateT<I, E>{first});
    add(firstp); // Appending to the main clause list.
  }
  if (!last.empty()) {
    auto *lastp = makeClause(llvm::omp::Clause::OMPC_lastprivate,
                             tomp::clause::LastprivateT<I, E>{last});
    add(lastp); // Appending to the main clause list.
  }
  return true;
}

// NOWAIT
template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::applyClause(
    const tomp::clause::NowaitT<I, E> &clause, const ClauseType *node) {
  if (applyToOutermost(node))
    return true;
  //llvm::errs() << "Cannot apply NOWAIT\n";
  return false;
}

template <typename I, typename E, typename H>
bool CompositeInfoBase<I, E, H>::split() {
  bool success = true;

  for (llvm::omp::Directive leaf : llvm::omp::getLeafConstructs(construct))
    leafs.push_back(DirectiveInfo<I, E>{leaf});

  for (const ClauseType *node : nodes)
    addClauseSymsToMap(*node, node);

  // First we need to apply LINEAR, because it can generate additional
  // FIRSTPRIVATE and LASTPRIVATE clauses that apply to the combined/
  // composite construct.
  // Collect them separately, because they may modify the clause list.
  llvm::SmallVector<const ClauseType *> linears;
  for (const ClauseType *node : nodes) {
    if (node->id == llvm::omp::Clause::OMPC_linear)
      linears.push_back(node);
  }
  for (const auto *node : linears) {
    success = success &&
              applyClause(std::get<tomp::clause::LinearT<I, E>>(node->u), node);
  }

  // ALLOCATE clauses need to be applied last since they need to see
  // which directives have data-privatizing clauses.
  auto skip = [](const ClauseType *node) {
    switch (node->id) {
    case llvm::omp::Clause::OMPC_allocate:
    case llvm::omp::Clause::OMPC_linear:
      return true;
    default:
      return false;
    }
  };

  // Apply (almost) all clauses.
  for (const ClauseType *node : nodes) {
    if (skip(node))
      continue;
    success =
        success &&
        std::visit([&](auto &&s) { return applyClause(s, node); }, node->u);
  }

  // Apply ALLOCATE.
  for (const ClauseType *node : nodes) {
    if (node->id != llvm::omp::Clause::OMPC_allocate)
      continue;
    success =
        success &&
        std::visit([&](auto &&s) { return applyClause(s, node); }, node->u);
  }

  return success;
}

} // namespace tomp

#endif // OPENMPCLAUSES_H
