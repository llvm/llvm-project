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

namespace clang {
namespace syntax {
static ArrayRef<syntax::Token> tokens(syntax::Node *N) {
  assert(N->isOriginal() && "tokens of modified nodes are not well-defined");
  if (auto *L = dyn_cast<syntax::Leaf>(N))
    return llvm::makeArrayRef(L->token(), 1);
  auto *T = cast<syntax::Tree>(N);
  return llvm::makeArrayRef(T->firstLeaf()->token(),
                            T->lastLeaf()->token() + 1);
}

class SyntaxTreeTest : public ::testing::Test,
                       public ::testing::WithParamInterface<TestClangConfig> {
protected:
  // Build a syntax tree for the code.
  syntax::TranslationUnit *buildTree(StringRef Code,
                                     const TestClangConfig &ClangConfig) {
    // FIXME: this code is almost the identical to the one in TokensTest. Share
    //        it.
    class BuildSyntaxTree : public ASTConsumer {
    public:
      BuildSyntaxTree(syntax::TranslationUnit *&Root,
                      std::unique_ptr<syntax::TokenBuffer> &TB,
                      std::unique_ptr<syntax::Arena> &Arena,
                      std::unique_ptr<syntax::TokenCollector> Tokens)
          : Root(Root), TB(TB), Arena(Arena), Tokens(std::move(Tokens)) {
        assert(this->Tokens);
      }

      void HandleTranslationUnit(ASTContext &Ctx) override {
        TB =
            std::make_unique<syntax::TokenBuffer>(std::move(*Tokens).consume());
        Tokens = nullptr; // make sure we fail if this gets called twice.
        Arena = std::make_unique<syntax::Arena>(Ctx.getSourceManager(),
                                                Ctx.getLangOpts(), *TB);
        Root = syntax::buildSyntaxTree(*Arena, *Ctx.getTranslationUnitDecl());
      }

    private:
      syntax::TranslationUnit *&Root;
      std::unique_ptr<syntax::TokenBuffer> &TB;
      std::unique_ptr<syntax::Arena> &Arena;
      std::unique_ptr<syntax::TokenCollector> Tokens;
    };

    class BuildSyntaxTreeAction : public ASTFrontendAction {
    public:
      BuildSyntaxTreeAction(syntax::TranslationUnit *&Root,
                            std::unique_ptr<syntax::TokenBuffer> &TB,
                            std::unique_ptr<syntax::Arena> &Arena)
          : Root(Root), TB(TB), Arena(Arena) {}

      std::unique_ptr<ASTConsumer>
      CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
        // We start recording the tokens, ast consumer will take on the result.
        auto Tokens =
            std::make_unique<syntax::TokenCollector>(CI.getPreprocessor());
        return std::make_unique<BuildSyntaxTree>(Root, TB, Arena,
                                                 std::move(Tokens));
      }

    private:
      syntax::TranslationUnit *&Root;
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

    Invocation = createInvocationFromCommandLine(ArgsCStr, Diags, FS);
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
    BuildSyntaxTreeAction Recorder(Root, this->TB, this->Arena);

    // Action could not be executed but the frontend didn't identify any errors
    // in the code ==> problem in setting up the action.
    if (!Compiler.ExecuteAction(Recorder) &&
        Diags->getClient()->getNumErrors() == 0) {
      ADD_FAILURE() << "failed to run the frontend";
      std::abort();
    }
    return Root;
  }

  ::testing::AssertionResult treeDumpEqual(StringRef Code, StringRef Tree) {
    SCOPED_TRACE(llvm::join(GetParam().getCommandLineArgs(), " "));

    auto *Root = buildTree(Code, GetParam());
    if (Diags->getClient()->getNumErrors() != 0) {
      return ::testing::AssertionFailure()
             << "Source file has syntax errors, they were printed to the test "
                "log";
    }
    std::string Actual = std::string(StringRef(Root->dump(*Arena)).trim());
    // EXPECT_EQ shows the diff between the two strings if they are different.
    EXPECT_EQ(Tree.trim().str(), Actual);
    if (Actual != Tree.trim().str()) {
      return ::testing::AssertionFailure();
    }
    return ::testing::AssertionSuccess();
  }

  // Adds a file to the test VFS.
  void addFile(StringRef Path, StringRef Contents) {
    if (!FS->addFile(Path, time_t(),
                     llvm::MemoryBuffer::getMemBufferCopy(Contents))) {
      ADD_FAILURE() << "could not add a file to VFS: " << Path;
    }
  }

  /// Finds the deepest node in the tree that covers exactly \p R.
  /// FIXME: implement this efficiently and move to public syntax tree API.
  syntax::Node *nodeByRange(llvm::Annotations::Range R, syntax::Node *Root) {
    ArrayRef<syntax::Token> Toks = tokens(Root);

    if (Toks.front().location().isFileID() &&
        Toks.back().location().isFileID() &&
        syntax::Token::range(*SourceMgr, Toks.front(), Toks.back()) ==
            syntax::FileRange(SourceMgr->getMainFileID(), R.Begin, R.End))
      return Root;

    auto *T = dyn_cast<syntax::Tree>(Root);
    if (!T)
      return nullptr;
    for (auto *C = T->firstChild(); C != nullptr; C = C->nextSibling()) {
      if (auto *Result = nodeByRange(R, C))
        return Result;
    }
    return nullptr;
  }

  // Data fields.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      new DiagnosticsEngine(new DiagnosticIDs, DiagOpts.get());
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS =
      new llvm::vfs::InMemoryFileSystem;
  IntrusiveRefCntPtr<FileManager> FileMgr =
      new FileManager(FileSystemOptions(), FS);
  IntrusiveRefCntPtr<SourceManager> SourceMgr =
      new SourceManager(*Diags, *FileMgr);
  std::shared_ptr<CompilerInvocation> Invocation;
  // Set after calling buildTree().
  std::unique_ptr<syntax::TokenBuffer> TB;
  std::unique_ptr<syntax::Arena> Arena;
};
} // namespace syntax
} // namespace clang
