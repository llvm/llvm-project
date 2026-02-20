//===- UnsafeBufferUsage.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGE_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGE_H

#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include "llvm/ADT/iterator_range.h"
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
/// integer indicating the pointer level: '(EntityId, PointerLevel)'.  An
/// EntityPointerLevel 'P' is valid iff
///   - 'P.EntityId' has a pointer type with at least 'P.PointerLevel' levels
///     (This implies 'P.PointerLevel > 0');
///   - 'P.EntityId' identifies an lvalue object and 'P.PointerLevel == 0'.
/// The latter case represents address-of expressions.
///
/// For the same example 'int *p[10];', the EntityPointerLevels below are valid:
/// '(p, 1)' is associated with 'int *[10]' of 'p';
/// '(p, 2)' is associated with 'int *' of 'p';
/// '(p, 0)' represents '&p'.
class EntityPointerLevel {
  EntityId Entity;
  unsigned PointerLevel;

  friend class UnsafeBufferUsageTUSummaryBuilder;
  friend class UnsafeBufferUsageEntitySummary;

  EntityPointerLevel(EntityId Entity, unsigned PointerLevel)
      : Entity(Entity), PointerLevel(PointerLevel) {}

public:
  EntityId getEntity() const { return Entity; }
  unsigned getPointerLevel() const { return PointerLevel; }

  bool operator==(const EntityPointerLevel &Other) const {
    return Entity == Other.Entity && PointerLevel == Other.PointerLevel;
  }

  bool operator!=(const EntityPointerLevel &Other) const {
    return !(*this == Other);
  }

  bool operator<(const EntityPointerLevel &Other) const {
    return std::tie(Entity, PointerLevel) <
           std::tie(Other.Entity, Other.PointerLevel);
  }

  // Comparator supporting partial comparison against EntityId:
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

  friend class UnsafeBufferUsageTUSummaryBuilder;

  UnsafeBufferUsageEntitySummary(EntityPointerLevelSet &&UnsafeBuffers)
      : EntitySummary(), UnsafeBuffers(std::move(UnsafeBuffers)) {}

public:
  using const_iterator = EntityPointerLevelSet::const_iterator;

  const_iterator begin() const { return UnsafeBuffers.begin(); }
  const_iterator end() const { return UnsafeBuffers.end(); }

  const_iterator find(const EntityPointerLevel &V) const {
    return UnsafeBuffers.find(V);
  }

  llvm::iterator_range<const_iterator> getSubsetOf(EntityId Entity) const {
    return llvm::make_range(UnsafeBuffers.equal_range(Entity));
  }

  /// \return the size of the set of EntityLevelPointers, which represents the
  /// set of unsafe buffers
  size_t getNumUnsafeBuffers() { return UnsafeBuffers.size(); }

  SummaryName getSummaryName() const override {
    return SummaryName{"UnsafeBufferUsage"};
  };
};
} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGE_H
