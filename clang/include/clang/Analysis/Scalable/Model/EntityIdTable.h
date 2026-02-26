//===- EntityIdTable.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITY_ID_TABLE_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITY_ID_TABLE_H

#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include <functional>
#include <map>

namespace clang::ssaf {

/// Manages entity name interning and provides efficient EntityId handles.
///
/// The table maps each unique EntityName to exactly one EntityId.
/// Entities are never removed.
class EntityIdTable {
  friend class SerializationFormat;
  friend class TestFixture;

  std::map<EntityName, EntityId> Entities;

public:
  EntityIdTable() = default;

  /// Creates or retrieves an EntityId for the given EntityName.
  ///
  /// If the entity already exists in the table, returns its existing Id.
  /// Otherwise, creates and returns a new Id. This operation is idempotent.
  EntityId getId(const EntityName &Name);

  /// Returns true if an entity with the given name exists in the table.
  bool contains(const EntityName &Name) const;

  /// Invokes the callback for each entity in the table.
  ///
  /// Iteration order is unspecified.
  void forEach(
      llvm::function_ref<void(const EntityName &, EntityId)> Callback) const;

  /// Returns the number of unique entities in the table.
  size_t count() const { return Entities.size(); }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITY_ID_TABLE_H
