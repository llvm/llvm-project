//===--- FeatureModuleExample.cpp - Example dynamic FeatureModule ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FeatureModule.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"

namespace clang::clangd {
namespace {

/// An example tweak contributed by the dynamic feature module.
class ExampleModuleTweak final : public Tweak {
public:
  const char *id() const override { return "ExampleModuleTweak"; }
  bool prepare(const Selection &) override { return true; }
  Expected<Effect> apply(const Selection &) override {
    return Effect::showMessage("Example module tweak executed");
  }
  std::string title() const override { return "Execute example module action"; }
  llvm::StringLiteral kind() const override {
    return llvm::StringLiteral("quickfix");
  }
};

/// An example FeatureModule loaded dynamically into clangd.
class ExampleFeatureModule final : public FeatureModule {
public:
  void contributeTweaks(std::vector<std::unique_ptr<Tweak>> &Out) override {
    Out.emplace_back(new ExampleModuleTweak);
  }
};

static FeatureModuleRegistry::Add<ExampleFeatureModule>
    X("example-feature-module", "Example clangd feature module");

} // namespace
} // namespace clang::clangd
