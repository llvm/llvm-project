//===--- Module.cpp - Plugging features into clangd -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Module.h"
#include "support/Logger.h"

namespace clang {
namespace clangd {

void Module::initialize(const Facilities &F) {
  assert(!Fac.hasValue() && "Initialized twice");
  Fac.emplace(F);
}

Module::Facilities &Module::facilities() {
  assert(Fac.hasValue() && "Not initialized yet");
  return *Fac;
}

bool ModuleSet::addImpl(void *Key, std::unique_ptr<Module> M,
                        const char *Source) {
  if (!Map.try_emplace(Key, M.get()).second) {
    // Source should (usually) include the name of the concrete module type.
    elog("Tried to register duplicate modules via {0}", Source);
    return false;
  }
  Modules.push_back(std::move(M));
  return true;
}

} // namespace clangd
} // namespace clang
