//===--- DummyFeatureModule.cpp -------------------------------------------===//
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

class Dummy final : public FeatureModule {
  static constexpr const char *TweakID = "DummyTweak";
  struct DummyTweak final : public Tweak {
    const char *id() const override { return TweakID; }
    bool prepare(const Selection &) override { return true; }
    Expected<Effect> apply(const Selection &) override {
      return error("not implemented");
    }
    std::string title() const override { return id(); }
    llvm::StringLiteral kind() const override {
      return llvm::StringLiteral("");
    };
  };

  void contributeTweaks(std::vector<std::unique_ptr<Tweak>> &Out) override {
    Out.emplace_back(new DummyTweak);
  }
};

static FeatureModuleRegistry::Add<Dummy>
    X("dummy", "Dummy feature module with dummy tweak");

// This anchor is used to force the linker to link in the generated object file
// and thus register the Dummy feature module.
volatile int DummyFeatureModuleAnchorSource = 0;

} // namespace clang::clangd
