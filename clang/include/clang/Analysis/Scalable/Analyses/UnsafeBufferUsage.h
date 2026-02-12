//===- UnsafeBufferUsage.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_H

#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummaryBuilder.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/iterator_range.h"
#include <memory>
#include <set>

namespace clang::ssaf {

/// A PointerKindVariable is associated with a pointer type as (a spelling part
/// of) the declared type of an entity. In other words, a PointerKindVariable
/// is associated with a `*` in the fully expanded spelling of the declared
/// type.
///
/// For example, for `int **p;`, there are two PointerKindVariables. One is
/// associated with `int **` and the other is associated with `int *`.
///
/// A PointerKindVariable can be identified by an EntityId, of which the
/// declared type is a pointer type, and an unsigned integer indicating the
/// pointer level with 1 referring to the whole declared pointer type.
///
/// For the same example `int **p;`, the two PointerKindVariables are:
/// '(p, 1)' for `int **` and '(p, 2)' for `int *`.
///
/// Reserve pointer level value 0 for representing address-of expressions in
/// implementations internally. For the same example, `&p` can be represented as
/// '(p, 0)'. With that, the PointerKindVariable derived from pointer expression
/// `*(&p)` or `&(*p)` is naturally equivalent to the one derived from
/// expression `p`.
class PointerKindVariable {
  EntityId Entity;
  unsigned PointerLevel;

  friend class UnsafeBufferUsageTUSummaryBuilder;
  friend class UnsafeBufferUsageEntitySummary;

  PointerKindVariable(EntityId Entity, unsigned PointerLevel)
      : Entity(Entity), PointerLevel(PointerLevel) {}

public:
  EntityId getEntity() const { return Entity; }
  unsigned getPointerLevel() const { return PointerLevel; }

  bool operator==(const PointerKindVariable &Other) const {
    return Entity == Other.Entity && PointerLevel == Other.PointerLevel;
  }

  bool operator!=(const PointerKindVariable &Other) const {
    return !(*this == Other);
  }

  bool operator<(const PointerKindVariable &Other) const {
    return std::tie(Entity, PointerLevel) <
           std::tie(Other.Entity, Other.PointerLevel);
  }

  // Comparator supporting partial comparison against EntityId:
  struct Comparator {
    using is_transparent = void;
    bool operator()(const PointerKindVariable &L,
                    const PointerKindVariable &R) const {
      return L < R;
    }
    bool operator()(const EntityId &L, const PointerKindVariable &R) const {
      return L < R.getEntity();
    }
    bool operator()(const PointerKindVariable &L, const EntityId &R) const {
      return L.getEntity() < R;
    }
  };
};

using PointerKindVariableSet =
    std::set<PointerKindVariable, PointerKindVariable::Comparator>;

/// An UnsafeBufferUsageEntitySummary is an immutable set of unsafe buffers, in
/// the form of PointerKindVariable.
class UnsafeBufferUsageEntitySummary final : public EntitySummary {
  const PointerKindVariableSet UnsafeBuffers;

  friend class UnsafeBufferUsageTUSummaryBuilder;

  UnsafeBufferUsageEntitySummary(PointerKindVariableSet &&UnsafeBuffers)
      : EntitySummary(), UnsafeBuffers(std::move(UnsafeBuffers)) {}

public:
  using const_iterator = PointerKindVariableSet::const_iterator;

  const_iterator begin() const { return UnsafeBuffers.begin(); }
  const_iterator end() const { return UnsafeBuffers.end(); }

  const_iterator find(const PointerKindVariable &V) const {
    return UnsafeBuffers.find(V);
  }

  llvm::iterator_range<const_iterator> getSubsetOf(EntityId Entity) const {
    return llvm::make_range(UnsafeBuffers.equal_range(Entity));
  }

  size_t getNumUnsafeBuffers() { return UnsafeBuffers.size(); }

  SummaryName getSummaryName() const override {
    return SummaryName{"UnsafeBufferUsage"};
  };
};

class UnsafeBufferUsageTUSummaryBuilder : public TUSummaryBuilder {
public:
  static PointerKindVariable buildPointerKindVariable(EntityId Entity,
                                                      unsigned PointerLevel) {
    return {Entity, PointerLevel};
  }

  static std::unique_ptr<UnsafeBufferUsageEntitySummary>
  buildUnsafeBufferUsageEntitySummary(PointerKindVariableSet &&UnsafeBuffers) {
    return std::make_unique<UnsafeBufferUsageEntitySummary>(
        UnsafeBufferUsageEntitySummary(std::move(UnsafeBuffers)));
  }
};
} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_H
