//===--- FeatureModulesTests.cpp  -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FeatureModule.h"
#include "Selection.h"
#include "TestTU.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang {
namespace clangd {
namespace {

TEST(FeatureModulesTest, ContributesTweak) {
  static constexpr const char *TweakID = "ModuleTweak";
  struct TweakContributingModule final : public FeatureModule {
    struct ModuleTweak final : public Tweak {
      const char *id() const override { return TweakID; }
      bool prepare(const Selection &Sel) override { return true; }
      Expected<Effect> apply(const Selection &Sel) override {
        return error("not implemented");
      }
      std::string title() const override { return id(); }
      llvm::StringLiteral kind() const override {
        return llvm::StringLiteral("");
      };
    };

    void contributeTweaks(std::vector<std::unique_ptr<Tweak>> &Out) override {
      Out.emplace_back(new ModuleTweak);
    }
  };

  FeatureModuleSet Set;
  Set.add(std::make_unique<TweakContributingModule>());

  auto AST = TestTU::withCode("").build();
  auto Tree =
      SelectionTree::createRight(AST.getASTContext(), AST.getTokens(), 0, 0);
  auto Actual = prepareTweak(
      TweakID, Tweak::Selection(nullptr, AST, 0, 0, std::move(Tree), nullptr),
      &Set);
  ASSERT_TRUE(bool(Actual));
  EXPECT_EQ(Actual->get()->id(), TweakID);
}

} // namespace
} // namespace clangd
} // namespace clang
