//===--- StandaloneTweakExample.cpp - Example standalone dynamic Tweak ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "refactor/Tweak.h"

namespace clang::clangd {
namespace {

/// A standalone tweak registered directly via REGISTER_TWEAK without a
/// FeatureModule.
class StandaloneTweak final : public Tweak {
public:
  const char *id() const override;
  bool prepare(const Selection &) override { return true; }
  Expected<Effect> apply(const Selection &) override {
    return Effect::showMessage("Standalone tweak executed");
  }
  std::string title() const override { return "Execute standalone action"; }
  llvm::StringLiteral kind() const override {
    return llvm::StringLiteral("quickfix");
  }
};

REGISTER_TWEAK(StandaloneTweak)

} // namespace
} // namespace clang::clangd
