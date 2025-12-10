//===- EntityIdTable.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include <algorithm>
#include <cassert>

namespace clang {
namespace ssaf {

EntityId EntityIdTable::createEntityId(const EntityName &Name) {
  auto [It, Inserted] = Entities.insert(Name);

  if (Inserted) {
    IdToEntity.push_back(&(*It));
    return EntityId(IdToEntity.size() - 1);
  }

  const EntityName *EntityPtr = &(*It);
  auto IdIt = std::find(IdToEntity.begin(), IdToEntity.end(), EntityPtr);
  assert(IdIt != IdToEntity.end() && "Entity exists but has no ID");

  size_t Index = std::distance(IdToEntity.begin(), IdIt);
  return EntityId(Index);
}

bool EntityIdTable::exists(const EntityName &Name) const {
  return Entities.find(Name) != Entities.end();
}

void EntityIdTable::forEach(
    std::function<void(const EntityName &, EntityId)> Callback) const {
  for (size_t Index = 0; Index < IdToEntity.size(); ++Index) {
    EntityId EId(Index);
    const EntityName &Name = *IdToEntity[Index];
    Callback(Name, EId);
  }
}

size_t EntityIdTable::count() const { return IdToEntity.size(); }

} // namespace ssaf
} // namespace clang
