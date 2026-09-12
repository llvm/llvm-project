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
#include "clang/AST/Decl.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <functional>
#include <memory>

namespace clang {
namespace clangd {
namespace {

struct TestModule final : FeatureModule {
  struct Listener final : ASTListener {
    Listener(TestModule &Module) : Module(Module) {}

    void beforePPCallbacks(CompilerInstance &CI) override {
      if (Module.BeforePPCallbacks)
        Module.BeforePPCallbacks(CI);
    }
    void beforeExecute(CompilerInstance &CI) override {
      if (Module.BeforeExecute)
        Module.BeforeExecute(CI);
    }
    void afterExecute(CompilerInstance &CI) override {
      if (Module.AfterExecute)
        Module.AfterExecute(CI);
    }
    void finalizeDiagnostic(clangd::Diag &Diag) override {
      if (Module.FinalizeDiagnostic)
        Module.FinalizeDiagnostic(Diag);
    }

  private:
    TestModule &Module;
  };

  std::unique_ptr<ASTListener> astListeners() override {
    return std::make_unique<Listener>(*this);
  }

  std::function<void(CompilerInstance &)> BeforePPCallbacks;
  std::function<void(CompilerInstance &)> BeforeExecute;
  std::function<void(CompilerInstance &)> AfterExecute;
  std::function<void(clangd::Diag &)> FinalizeDiagnostic;
};

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
    EXPECT_THAT(AST.getDiagnostics(), testing::Not(testing::IsEmpty()));
  }

  TU.FeatureModules = &FMS;
  {
    auto AST = TU.build();
    EXPECT_THAT(AST.getDiagnostics(), testing::IsEmpty());
  }
}

TEST(FeatureModulesTest, BeforePPCallbacks) {
  struct IncludeRecorder : public PPCallbacks {
    IncludeRecorder(std::vector<std::string> &Includes) : Includes(Includes) {}

    void InclusionDirective(SourceLocation, const Token &, StringRef FileName,
                            bool, CharSourceRange, OptionalFileEntryRef,
                            StringRef, StringRef, const clang::Module *, bool,
                            SrcMgr::CharacteristicKind) override {
      Includes.push_back(FileName.str());
    }

  private:
    std::vector<std::string> &Includes;
  };
  std::vector<std::string> Includes;
  auto Module = std::make_unique<TestModule>();
  Module->BeforePPCallbacks = [&Includes](CompilerInstance &CI) {
    // The preamble build sees this include directly. Register only during the
    // main-file build to verify the callback sees the replayed event.
    if (CI.getFrontendOpts().ProgramAction == frontend::ParseSyntaxOnly)
      CI.getPreprocessor().addPPCallbacks(
          std::make_unique<IncludeRecorder>(Includes));
  };
  FeatureModuleSet FMS;
  FMS.add(std::move(Module));

  TestTU TU = TestTU::withCode(R"cpp(
    #include "header.h"
  )cpp");
  TU.AdditionalFiles["header.h"] = "";
  TU.FeatureModules = &FMS;
  TU.build();
  EXPECT_THAT(Includes, testing::ElementsAre("header.h"));
}

TEST(FeatureModulesTest, BeforeExecute) {
  auto Module = std::make_unique<TestModule>();
  Module->BeforeExecute = [](CompilerInstance &CI) {
    CI.getPreprocessor().SetSuppressIncludeNotFoundError(true);
  };
  FeatureModuleSet FMS;
  FMS.add(std::move(Module));

  TestTU TU = TestTU::withCode(R"cpp(
    /*error-ok*/
    #include "not_found.h"

    void foo() {
      #include "not_found_not_preamble.h"
    }
  )cpp");

  {
    auto AST = TU.build();
    EXPECT_THAT(AST.getDiagnostics(), testing::Not(testing::IsEmpty()));
  }

  TU.FeatureModules = &FMS;
  {
    auto AST = TU.build();
    EXPECT_THAT(AST.getDiagnostics(), testing::IsEmpty());
  }
}

TEST(FeatureModulesTest, AfterExecute) {
  std::vector<std::string> DeclNames;
  auto Module = std::make_unique<TestModule>();
  Module->AfterExecute = [&DeclNames](CompilerInstance &CI) {
    for (Decl *D : CI.getASTContext().getTraversalScope())
      if (const auto *ND = llvm::dyn_cast<NamedDecl>(D))
        DeclNames.push_back(ND->getNameAsString());
  };
  FeatureModuleSet FMS;
  FMS.add(std::move(Module));

  TestTU TU = TestTU::withCode(R"cpp(
    #include "header.h"
    void mainFileFunc();
  )cpp");
  TU.AdditionalFiles["header.h"] = "void headerFunc();";
  TU.FeatureModules = &FMS;
  TU.build();

  // afterExecute runs once clangd has restricted the traversal scope, so the
  // declaration from the header is intentionally not visible here.
  EXPECT_THAT(DeclNames, testing::ElementsAre("mainFileFunc"));
}

TEST(FeatureModulesTest, FinalizeDiagnostic) {
  unsigned Notes = 0;
  unsigned Fixes = 0;
  auto Module = std::make_unique<TestModule>();
  Module->FinalizeDiagnostic = [&](clangd::Diag &Diag) {
    if (Diag.Message.find("undeclared identifier 'fooo'") == std::string::npos)
      return;
    Notes = Diag.Notes.size();
    Fixes = Diag.Fixes.size();
  };
  FeatureModuleSet FMS;
  FMS.add(std::move(Module));

  TestTU TU = TestTU::withCode(R"cpp(
    void foo();
    void bar() { fooo(); } // error-ok
  )cpp");
  TU.FeatureModules = &FMS;
  EXPECT_THAT(TU.build().getDiagnostics(), testing::SizeIs(1));
  EXPECT_EQ(Notes, 1u);
  EXPECT_EQ(Fixes, 1u);
}

} // namespace
} // namespace clangd
} // namespace clang
