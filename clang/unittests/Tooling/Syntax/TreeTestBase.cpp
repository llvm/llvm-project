//===- TreeTestBase.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the test infrastructure for syntax trees.
//
//===----------------------------------------------------------------------===//

#include "TreeTestBase.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Testing/CommandLineArgs.h"
#include "clang/Testing/TestClangConfig.h"
#include "clang/Tooling/Syntax/BuildTree.h"
#include "clang/Tooling/Syntax/Nodes.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "clang/Tooling/Syntax/Tree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::syntax;

namespace {
ArrayRef<syntax::Token> tokens(syntax::Node *N,
                               const TokenBufferTokenManager &STM) {
  assert(N->isOriginal() && "tokens of modified nodes are not well-defined");
  if (auto *L = dyn_cast<syntax::Leaf>(N))
    return llvm::makeArrayRef(STM.getToken(L->getTokenKey()), 1);
  auto *T = cast<syntax::Tree>(N);
  return llvm::makeArrayRef(STM.getToken(T->findFirstLeaf()->getTokenKey()),
                            STM.getToken(T->findLastLeaf()->getTokenKey()) + 1);
}
} // namespace

std::vector<TestClangConfig> clang::syntax::allTestClangConfigs() {
  std::vector<TestClangConfig> all_configs;
  for (TestLanguage lang : {Lang_C89, Lang_C99, Lang_CXX03, Lang_CXX11,
                            Lang_CXX14, Lang_CXX17, Lang_CXX20}) {
    TestClangConfig config;
    config.Language = lang;
    config.Target = "x86_64-pc-linux-gnu";
    all_configs.push_back(config);

    // Windows target is interesting to test because it enables
    // `-fdelayed-template-parsing`.
    config.Target = "x86_64-pc-win32-msvc";
    all_configs.push_back(config);
  }
  return all_configs;
}

syntax::TranslationUnit *
SyntaxTreeTest::buildTree(StringRef Code, const TestClangConfig &ClangConfig) {
  // FIXME: this code is almost the identical to the one in TokensTest. Share
  //        it.
  class BuildSyntaxTree : public ASTConsumer {
  public:
    BuildSyntaxTree(syntax::TranslationUnit *&Root,
                    std::unique_ptr<syntax::TokenBuffer> &TB,
                    std::unique_ptr<syntax::TokenBufferTokenManager> &TM,
                    std::unique_ptr<syntax::Arena> &Arena,
                    std::unique_ptr<syntax::TokenCollector> Tokens)
        : Root(Root), TB(TB), TM(TM), Arena(Arena), Tokens(std::move(Tokens)) {
      assert(this->Tokens);
    }

    void HandleTranslationUnit(ASTContext &Ctx) override {
      TB = std::make_unique<syntax::TokenBuffer>(std::move(*Tokens).consume());
      Tokens = nullptr; // make sure we fail if this gets called twice.
      TM = std::make_unique<syntax::TokenBufferTokenManager>(
          *TB, Ctx.getLangOpts(), Ctx.getSourceManager());
      Arena = std::make_unique<syntax::Arena>();
      Root = syntax::buildSyntaxTree(*Arena, *TM, Ctx);
    }

  private:
    syntax::TranslationUnit *&Root;
    std::unique_ptr<syntax::TokenBuffer> &TB;
    std::unique_ptr<syntax::TokenBufferTokenManager> &TM;
    std::unique_ptr<syntax::Arena> &Arena;
    std::unique_ptr<syntax::TokenCollector> Tokens;
  };

  class BuildSyntaxTreeAction : public ASTFrontendAction {
  public:
    BuildSyntaxTreeAction(syntax::TranslationUnit *&Root,
                          std::unique_ptr<syntax::TokenBufferTokenManager> &TM,
                          std::unique_ptr<syntax::TokenBuffer> &TB,
                          std::unique_ptr<syntax::Arena> &Arena)
        : Root(Root), TM(TM), TB(TB), Arena(Arena) {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef InFile) override {
      // We start recording the tokens, ast consumer will take on the result.
      auto Tokens =
          std::make_unique<syntax::TokenCollector>(CI.getPreprocessor());
      return std::make_unique<BuildSyntaxTree>(Root, TB, TM, Arena,
                                               std::move(Tokens));
    }

  private:
    syntax::TranslationUnit *&Root;
    std::unique_ptr<syntax::TokenBufferTokenManager> &TM;
    std::unique_ptr<syntax::TokenBuffer> &TB;
    std::unique_ptr<syntax::Arena> &Arena;
  };

  constexpr const char *FileName = "./input.cpp";
  FS->addFile(FileName, time_t(), llvm::MemoryBuffer::getMemBufferCopy(""));

  if (!Diags->getClient())
    Diags->setClient(new TextDiagnosticPrinter(llvm::errs(), DiagOpts.get()));
  Diags->setSeverityForGroup(diag::Flavor::WarningOrError, "unused-value",
                             diag::Severity::Ignored, SourceLocation());

  // Prepare to run a compiler.
  std::vector<std::string> Args = {
      "syntax-test",
      "-fsyntax-only",
  };
  llvm::copy(ClangConfig.getCommandLineArgs(), std::back_inserter(Args));
  Args.push_back(FileName);

  std::vector<const char *> ArgsCStr;
  for (const std::string &arg : Args) {
    ArgsCStr.push_back(arg.c_str());
  }

  CreateInvocationOptions CIOpts;
  CIOpts.Diags = Diags;
  CIOpts.VFS = FS;
  Invocation = createInvocation(ArgsCStr, std::move(CIOpts));
  assert(Invocation);
  Invocation->getFrontendOpts().DisableFree = false;
  Invocation->getPreprocessorOpts().addRemappedFile(
      FileName, llvm::MemoryBuffer::getMemBufferCopy(Code).release());
  CompilerInstance Compiler;
  Compiler.setInvocation(Invocation);
  Compiler.setDiagnostics(Diags.get());
  Compiler.setFileManager(FileMgr.get());
  Compiler.setSourceManager(SourceMgr.get());

  syntax::TranslationUnit *Root = nullptr;
  BuildSyntaxTreeAction Recorder(Root, this->TM, this->TB, this->Arena);

  // Action could not be executed but the frontend didn't identify any errors
  // in the code ==> problem in setting up the action.
  if (!Compiler.ExecuteAction(Recorder) &&
      Diags->getClient()->getNumErrors() == 0) {
    ADD_FAILURE() << "failed to run the frontend";
    std::abort();
  }
  return Root;
}

syntax::Node *SyntaxTreeTest::nodeByRange(llvm::Annotations::Range R,
                                          syntax::Node *Root) {
  ArrayRef<syntax::Token> Toks = tokens(Root, *TM);

  if (Toks.front().location().isFileID() && Toks.back().location().isFileID() &&
      syntax::Token::range(*SourceMgr, Toks.front(), Toks.back()) ==
          syntax::FileRange(SourceMgr->getMainFileID(), R.Begin, R.End))
    return Root;

  auto *T = dyn_cast<syntax::Tree>(Root);
  if (!T)
    return nullptr;
  for (auto *C = T->getFirstChild(); C != nullptr; C = C->getNextSibling()) {
    if (auto *Result = nodeByRange(R, C))
      return Result;
  }
  return nullptr;
}
