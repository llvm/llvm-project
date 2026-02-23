//===- EntityId.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the EntityId class, which provides a lightweight opaque
// handle to entities in an EntityIdTable. EntityIds are index-based for
// efficient comparison and lookup.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITY_ID_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITY_ID_H

#include "llvm/Support/raw_ostream.h"
#include <cstddef>

namespace clang::ssaf {

class EntityIdTable;

/// Lightweight opaque handle representing an entity in an EntityIdTable.
///
/// EntityIds are created by EntityIdTable. Equality and ordering comparisons
/// are well-defined for EntityIds created by the same EntityIdTable.
///
/// \see EntityIdTable
class EntityId {
  friend class EntityIdTable;
  friend class SerializationFormat;
  friend class TestFixture;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const EntityId &Id);

  size_t Index;

  explicit EntityId(size_t Index) : Index(Index) {}

  EntityId() = delete;

public:
  bool operator==(const EntityId &Other) const { return Index == Other.Index; }
  bool operator<(const EntityId &Other) const { return Index < Other.Index; }
  bool operator!=(const EntityId &Other) const { return !(*this == Other); }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const EntityId &Id);

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITY_ID_H
