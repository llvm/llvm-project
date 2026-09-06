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
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <array>
#include <memory>
#include <optional>

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
  struct PPCallbackModule final : public FeatureModule {
    struct Listener : public FeatureModule::ASTListener {
      Listener(std::vector<std::string> &Includes) : Includes(Includes) {}

      void beforePPCallbacks(CompilerInstance &CI) override {
        // The preamble build sees this include directly. Register only during
        // the main-file build to verify the callback sees the replayed event.
        if (CI.getFrontendOpts().ProgramAction == frontend::ParseSyntaxOnly)
          CI.getPreprocessor().addPPCallbacks(
              std::make_unique<IncludeRecorder>(Includes));
      }

    private:
      std::vector<std::string> &Includes;
    };

    PPCallbackModule(std::vector<std::string> &Includes) : Includes(Includes) {}
    std::unique_ptr<ASTListener> astListeners() override {
      return std::make_unique<Listener>(Includes);
    };

  private:
    std::vector<std::string> &Includes;
  };

  std::vector<std::string> Includes;
  FeatureModuleSet FMS;
  FMS.add(std::make_unique<PPCallbackModule>(Includes));

  TestTU TU = TestTU::withCode(R"cpp(
    #include "header.h"
  )cpp");
  TU.AdditionalFiles["header.h"] = "";
  TU.FeatureModules = &FMS;
  TU.build();
  EXPECT_THAT(Includes, testing::ElementsAre("header.h"));
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
    EXPECT_THAT(AST.getDiagnostics(), testing::Not(testing::IsEmpty()));
  }

  TU.FeatureModules = &FMS;
  {
    auto AST = TU.build();
    EXPECT_THAT(AST.getDiagnostics(), testing::IsEmpty());
  }
}

TEST(FeatureModulesTest, AfterExecute) {
  struct AfterExecuteState {
    bool ReenterPreprocessorInConsumer = false;
    bool HandledTranslationUnit = false;
    bool ConsumerSawWholeTranslationUnit = false;
    bool AfterExecuteCalled = false;
    bool ReenteredPreprocessor = false;
    std::vector<std::string> DeclNames;
  };
  struct AfterExecuteModule final : public FeatureModule {
    struct Consumer : public ASTConsumer {
      Consumer(AfterExecuteState &S, CompilerInstance &CI) : S(S), CI(CI) {}

      void HandleTranslationUnit(ASTContext &Ctx) override {
        S.HandledTranslationUnit = true;
        auto Scope = Ctx.getTraversalScope();
        S.ConsumerSawWholeTranslationUnit =
            Scope.size() == 1 && Scope.front() == Ctx.getTranslationUnitDecl();
        Pending = &Ctx;
        if (S.ReenterPreprocessorInConsumer)
          reenterAtEOF(CI);
      }

      void run(CompilerInstance &CI) {
        if (!Pending)
          return;
        SourceLocation MainFileDeclLoc;
        for (Decl *D : Pending->getTraversalScope()) {
          if (const auto *ND = llvm::dyn_cast<NamedDecl>(D)) {
            S.DeclNames.push_back(ND->getNameAsString());
            MainFileDeclLoc = ND->getLocation();
          }
        }

        if (MainFileDeclLoc.isValid())
          S.ReenteredPreprocessor = reenterPreprocessor(CI, MainFileDeclLoc);
      }

    private:
      static void reenterAtEOF(CompilerInstance &CI) {
        Token End;
        End.startToken();
        auto &SM = CI.getSourceManager();
        End.setLocation(SM.getLocForEndOfFile(SM.getMainFileID()));
        End.setKind(tok::eof);
        std::array<Token, 1> Stream{End};
        auto &PP = CI.getPreprocessor();
        PP.EnterTokenStream(Stream, /*DisableMacroExpansion=*/false,
                            /*IsReinject=*/false);
        PP.Lex(End);
      }

      static bool reenterPreprocessor(CompilerInstance &CI,
                                      SourceLocation MainFileDeclLoc) {
        Token Reinjected;
        if (Lexer::getRawToken(MainFileDeclLoc, Reinjected,
                               CI.getSourceManager(), CI.getLangOpts()))
          return false;
        auto &PP = CI.getPreprocessor();
        PP.LookUpIdentifierInfo(Reinjected);
        Token End;
        End.startToken();
        End.setKind(tok::eof);
        std::array<Token, 2> Stream{Reinjected, End};
        PP.EnterTokenStream(Stream, /*DisableMacroExpansion=*/false,
                            /*IsReinject=*/false);
        do {
          PP.Lex(Reinjected);
        } while (Reinjected.isNot(tok::eof));
        return true;
      }

      AfterExecuteState &S;
      CompilerInstance &CI;
      ASTContext *Pending = nullptr;
    };

    struct Listener : public FeatureModule::ASTListener {
      Listener(AfterExecuteState &S) : S(S) {}

      void beforeExecute(CompilerInstance &CI) override {
        std::vector<std::unique_ptr<ASTConsumer>> Consumers;
        Consumers.push_back(CI.takeASTConsumer());
        auto Deferred = std::make_unique<Consumer>(S, CI);
        DeferredConsumer = Deferred.get();
        Consumers.push_back(std::move(Deferred));
        CI.setASTConsumer(
            std::make_unique<MultiplexConsumer>(std::move(Consumers)));
      }

      void afterExecute(CompilerInstance &CI) override {
        S.AfterExecuteCalled = true;
        if (DeferredConsumer)
          DeferredConsumer->run(CI);
      }

    private:
      AfterExecuteState &S;
      Consumer *DeferredConsumer = nullptr;
    };

    AfterExecuteModule(AfterExecuteState &S) : S(S) {}
    std::unique_ptr<ASTListener> astListeners() override {
      return std::make_unique<Listener>(S);
    };

  private:
    AfterExecuteState &S;
  };

  // HandleTranslationUnit is too early for work that re-enters the
  // preprocessor: clangd's token collector is still installed, observes the
  // extra token, and cannot build a valid TokenBuffer afterwards.
  AfterExecuteState EarlyState;
  EarlyState.ReenterPreprocessorInConsumer = true;
  FeatureModuleSet EarlyFMS;
  EarlyFMS.add(std::make_unique<AfterExecuteModule>(EarlyState));
  TestTU EarlyTU = TestTU::withCode("int mainFileFunc();");
  EarlyTU.FeatureModules = &EarlyFMS;
  EXPECT_DEATH_IF_SUPPORTED((void)EarlyTU.build(),
                            "Couldn't map expanded token");

  AfterExecuteState State;
  FeatureModuleSet FMS;
  FMS.add(std::make_unique<AfterExecuteModule>(State));

  TestTU TU = TestTU::withCode(R"cpp(
    #include "header.h"
    inline int mainFileFunc() { return 0; }
  )cpp");
  TU.AdditionalFiles["header.h"] = "void headerFunc();";
  TU.FeatureModules = &FMS;
  auto AST = TU.build();
  // The multiplexed consumer runs before clangd replaces the whole-TU
  // traversal scope with the declarations originating in the main file. It
  // therefore cannot observe the finalized scope used by afterExecute below.
  EXPECT_TRUE(State.HandledTranslationUnit);
  EXPECT_TRUE(State.ConsumerSawWholeTranslationUnit);
  EXPECT_TRUE(State.AfterExecuteCalled);
  EXPECT_TRUE(State.ReenteredPreprocessor);

  // afterExecute runs once clangd has restricted the traversal scope, so the
  // declaration from the header is intentionally not visible here.
  EXPECT_THAT(State.DeclNames, testing::ElementsAre("mainFileFunc"));

  // The deferred preprocessing does not affect the token buffer: every parsed
  // main-file token is still present exactly once and in source order.
  std::vector<std::string> Tokens;
  for (const auto &Tok : AST.getTokens().expandedTokens())
    if (Tok.kind() != tok::eof)
      Tokens.push_back(Tok.text(AST.getSourceManager()).str());
  EXPECT_THAT(Tokens, testing::ElementsAre("inline", "int", "mainFileFunc", "(",
                                           ")", "{", "return", "0", ";", "}"));
}

TEST(FeatureModulesTest, FinalizeDiagnostic) {
  struct DiagnosticState {
    std::optional<clangd::Diag> AtSawDiagnostic;
    std::optional<clangd::Diag> AtFinalization;
  } State;
  struct DiagnosticModule final : public FeatureModule {
    struct Listener : public FeatureModule::ASTListener {
      Listener(DiagnosticState &State) : State(State) {}

      void sawDiagnostic(const clang::Diagnostic &,
                         clangd::Diag &Diag) override {
        if (Diag.Message.find("undeclared identifier 'fooo'") ==
            std::string::npos)
          return;
        State.AtSawDiagnostic = Diag;
      }

      void finalizeDiagnostic(clangd::Diag &Diag) override {
        if (Diag.Message.find("undeclared identifier 'fooo'") ==
            std::string::npos)
          return;
        State.AtFinalization = Diag;
      }

    private:
      DiagnosticState &State;
    };

    DiagnosticModule(DiagnosticState &State) : State(State) {}
    std::unique_ptr<ASTListener> astListeners() override {
      return std::make_unique<Listener>(State);
    };

  private:
    DiagnosticState &State;
  };
  FeatureModuleSet FMS;
  FMS.add(std::make_unique<DiagnosticModule>(State));

  TestTU TU = TestTU::withCode(R"cpp(
    void foo();
    void bar() { fooo(); } // error-ok
  )cpp");
  TU.FeatureModules = &FMS;
  EXPECT_THAT(TU.build().getDiagnostics(), testing::SizeIs(1));
  // sawDiagnostic runs as soon as clangd creates the primary diagnostic. The
  // subsequent note and the typo correction have not been attached yet.
  ASSERT_TRUE(State.AtSawDiagnostic);
  EXPECT_THAT(State.AtSawDiagnostic->Notes, testing::IsEmpty());
  EXPECT_THAT(State.AtSawDiagnostic->Fixes, testing::IsEmpty());

  // finalizeDiagnostic sees the assembled diagnostic after clangd has
  // associated its note and fix with the primary diagnostic.
  ASSERT_TRUE(State.AtFinalization);
  EXPECT_THAT(State.AtFinalization->Notes, testing::SizeIs(1));
  EXPECT_THAT(State.AtFinalization->Fixes, testing::SizeIs(1));
}

} // namespace
} // namespace clangd
} // namespace clang
