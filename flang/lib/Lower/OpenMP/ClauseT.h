//===- ClauseT -- clause template definitions -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_LOWER_OPENMP_CLAUSET_H
#define FORTRAN_LOWER_OPENMP_CLAUSET_H

#include "flang/Parser/parse-tree.h" // For enum reuse

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

#include "llvm/Frontend/OpenMP/OMP.h.inc"

namespace tomp {

template <typename T>
using ListT = llvm::SmallVector<T, 0>;

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
  using IntrinsicOperator = Fortran::parser::DefinedOperator::IntrinsicOperator;
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
  using OmpAtomicDefaultMemOrderType =
      Fortran::common::OmpAtomicDefaultMemOrderType;
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
  using ImplicitBehavior =
      Fortran::parser::OmpDefaultmapClause::ImplicitBehavior;
  using VariableCategory =
      Fortran::parser::OmpDefaultmapClause::VariableCategory;
  using TupleTrait = std::true_type;
  std::tuple<ImplicitBehavior, std::optional<VariableCategory>> t;
};

template <typename I, typename E>
struct DefaultT {
  using Type = Fortran::parser::OmpDefaultClause::Type;
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
  using Type = Fortran::parser::OmpDependenceType::Type;
  struct InOut {
    using TupleTrait = std::true_type;
    std::tuple<Type, ObjectListT<I, E>> t;
  };
  using UnionTrait = std::true_type;
  std::variant<Source, Sink, InOut> u;
};

template <typename I, typename E>
struct DeviceT {
  using DeviceModifier = Fortran::parser::OmpDeviceClause::DeviceModifier;
  using TupleTrait = std::true_type;
  std::tuple<std::optional<DeviceModifier>, E> t;
};

template <typename I, typename E>
struct DeviceTypeT {
  using Type = Fortran::parser::OmpDeviceTypeClause::Type;
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
  using DirectiveNameModifier =
      Fortran::parser::OmpIfClause::DirectiveNameModifier;
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
    using Type = Fortran::parser::OmpLinearModifier::Type;
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
    using Type = Fortran::parser::OmpMapType::Type;
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
  using Kind = Fortran::parser::OmpOrderModifier::Kind;
  using Type = Fortran::parser::OmpOrderClause::Type;
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
  using Type = Fortran::parser::OmpProcBindClause::Type;
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
  using ModType = Fortran::parser::OmpScheduleModifierType::ModType;
  struct ScheduleModifier {
    using TupleTrait = std::true_type;
    std::tuple<ModType, std::optional<ModType>> t;
  };
  using ScheduleType = Fortran::parser::OmpScheduleClause::ScheduleType;
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

} // namespace tomp

#endif // FORTRAN_LOWER_OPENMP_CLAUSET_H
