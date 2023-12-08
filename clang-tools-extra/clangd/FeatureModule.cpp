//===--- FeatureModule.cpp - Plugging features into clangd ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FeatureModule.h"
#include "support/Logger.h"

namespace clang {
namespace clangd {

void FeatureModule::initialize(const Facilities &F) {
  assert(!Fac && "Initialized twice");
  Fac.emplace(F);
}

FeatureModule::Facilities &FeatureModule::facilities() {
  assert(Fac && "Not initialized yet");
  return *Fac;
}

bool FeatureModuleSet::addImpl(void *Key, std::unique_ptr<FeatureModule> M,
                               const char *Source) {
  if (!Map.try_emplace(Key, M.get()).second) {
    // Source should (usually) include the name of the concrete module type.
    elog("Tried to register duplicate feature modules via {0}", Source);
    return false;
  }
  Modules.push_back(std::move(M));
  return true;
}

} // namespace clangd
} // namespace clang
