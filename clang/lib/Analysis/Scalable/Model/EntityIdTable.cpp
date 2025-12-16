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
  EntityId Id(Entities.size());
  const auto Res = Entities.try_emplace(Name, Id);
  return Res.first->second;
}

bool EntityIdTable::exists(const EntityName &Name) const {
  return Entities.find(Name) != Entities.end();
}

void EntityIdTable::forEach(
    llvm::function_ref<void(const EntityName &, EntityId)> Callback) const {
  for (const auto& [Name, Id] : Entities) {
    Callback(Name, Id);
  }
}

size_t EntityIdTable::count() const { return Entities.size(); }

} // namespace clang::ssaf
