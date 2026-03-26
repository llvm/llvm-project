//===- UnsafeBufferUsage.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGE_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGE_H

#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include <set>

namespace clang::ssaf {

/// An EntityPointerLevel represents a level of the declared pointer/array
/// type of an entity.  In the fully-expanded spelling of the declared type, a
/// EntityPointerLevel is associated with a '*' (or a '[]`) in that declaration.
///
/// For example, for 'int *p[10];', there are two EntityPointerLevels. One
/// is associated with 'int *[10]' of 'p' and the other is associated with 'int
/// *' of 'p'.
///
/// An EntityPointerLevel can be identified by an EntityId and an unsigned
/// integer indicating the pointer level: '(EntityId, PointerLevel)'.
/// An EntityPointerLevel 'P' is valid iff 'P.EntityId' has a pointer type with
/// at least 'P.PointerLevel' levels (This implies 'P.PointerLevel > 0').
///
/// For the same example 'int *p[10];', the EntityPointerLevels below are valid:
/// - '(p, 2)' is associated with the 'int *' part of the declared type of 'p';
/// - '(p, 1)' is associated with the 'int *[10]' part of the declared type of
/// 'p'.
class EntityPointerLevel {
  EntityId Entity;
  unsigned PointerLevel;

  friend class UnsafeBufferUsageTUSummaryExtractor;

  EntityPointerLevel(EntityId Entity, unsigned PointerLevel)
      : Entity(Entity), PointerLevel(PointerLevel) {}

public:
  EntityId getEntity() const { return Entity; }
  unsigned getPointerLevel() const { return PointerLevel; }

  bool operator==(const EntityPointerLevel &Other) const {
    return std::tie(Entity, PointerLevel) ==
           std::tie(Other.Entity, Other.PointerLevel);
  }

  bool operator!=(const EntityPointerLevel &Other) const {
    return !(*this == Other);
  }

  bool operator<(const EntityPointerLevel &Other) const {
    return std::tie(Entity, PointerLevel) <
           std::tie(Other.Entity, Other.PointerLevel);
  }

  /// Compares `EntityPointerLevel`s; additionally, partially compares
  /// `EntityPointerLevel` with `EntityId`.
  struct Comparator {
    using is_transparent = void;
    bool operator()(const EntityPointerLevel &L,
                    const EntityPointerLevel &R) const {
      return L < R;
    }
    bool operator()(const EntityId &L, const EntityPointerLevel &R) const {
      return L < R.getEntity();
    }
    bool operator()(const EntityPointerLevel &L, const EntityId &R) const {
      return L.getEntity() < R;
    }
  };
};

using EntityPointerLevelSet =
    std::set<EntityPointerLevel, EntityPointerLevel::Comparator>;

/// An UnsafeBufferUsageEntitySummary is an immutable set of unsafe buffers, in
/// the form of EntityPointerLevel.
class UnsafeBufferUsageEntitySummary final : public EntitySummary {
  const EntityPointerLevelSet UnsafeBuffers;

  friend class UnsafeBufferUsageTUSummaryExtractor;

  UnsafeBufferUsageEntitySummary(EntityPointerLevelSet UnsafeBuffers)
      : EntitySummary(), UnsafeBuffers(std::move(UnsafeBuffers)) {}

public:
  SummaryName getSummaryName() const override {
    return SummaryName{"UnsafeBufferUsage"};
  };

  bool operator==(const EntityPointerLevelSet &Other) const {
    return UnsafeBuffers == Other;
  }

  bool empty() const { return UnsafeBuffers.empty(); }
};
} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGE_H
