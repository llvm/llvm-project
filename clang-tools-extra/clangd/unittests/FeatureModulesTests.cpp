//===--- FeatureModulesTests.cpp  -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "FeatureModule.h"
#include "Selection.h"
#include "TestTU.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/Lex/PreprocessorOptions.h"
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

TEST(FeatureModulesTest, SuppressDiags) {
  struct DiagModifierModule final : public FeatureModule {
    struct Listener : public FeatureModule::ASTListener {
      void sawDiagnostic(const clang::Diagnostic &Info,
                         clangd::Diag &Diag) override {
        Diag.Severity = DiagnosticsEngine::Ignored;
      }
    };
    std::unique_ptr<ASTListener> astListeners() override {
      return std::make_unique<Listener>();
    };
  };
  FeatureModuleSet FMS;
  FMS.add(std::make_unique<DiagModifierModule>());

  Annotations Code("[[test]]; /* error-ok */");
  TestTU TU;
  TU.Code = Code.code().str();

  {
    auto AST = TU.build();
    EXPECT_THAT(*AST.getDiagnostics(), testing::Not(testing::IsEmpty()));
  }

  TU.FeatureModules = &FMS;
  {
    auto AST = TU.build();
    EXPECT_THAT(*AST.getDiagnostics(), testing::IsEmpty());
  }
}

TEST(FeatureModulesTest, BeforeExecute) {
  struct BeforeExecuteModule final : public FeatureModule {
    struct Listener : public FeatureModule::ASTListener {
      void beforeExecute(CompilerInstance &CI) override {
        CI.getPreprocessor().SetSuppressIncludeNotFoundError(true);
      }
    };
    std::unique_ptr<ASTListener> astListeners() override {
      return std::make_unique<Listener>();
    };
  };
  FeatureModuleSet FMS;
  FMS.add(std::make_unique<BeforeExecuteModule>());

  TestTU TU = TestTU::withCode(R"cpp(
    /*error-ok*/
    #include "not_found.h"

    void foo() {
      #include "not_found_not_preamble.h"
    }
  )cpp");

  {
    auto AST = TU.build();
    EXPECT_THAT(*AST.getDiagnostics(), testing::Not(testing::IsEmpty()));
  }

  TU.FeatureModules = &FMS;
  {
    auto AST = TU.build();
    EXPECT_THAT(*AST.getDiagnostics(), testing::IsEmpty());
  }
}

} // namespace
} // namespace clangd
} // namespace clang
