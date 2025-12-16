//===- EntityIdTable.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include <cassert>

namespace clang::ssaf {

EntityId EntityIdTable::getId(const EntityName &Name) {
  const auto It = Entities.find(Name);
  if (It == Entities.end()) {
    EntityId Id(Entities.size());
    Entities.emplace(Name, Id);
    return Id;
  }

  return It->second;
}

bool EntityIdTable::exists(const EntityName &Name) const {
  return Entities.find(Name) != Entities.end();
}

void EntityIdTable::forEach(
    std::function<void(const EntityName &, EntityId)> Callback) const {
  for (const auto& [Name, Id] : Entities) {
    Callback(Name, Id);
  }
}

size_t EntityIdTable::count() const { return Entities.size(); }

} // namespace clang::ssaf
