//===-- MlirCastRefactor.cpp - mlir refactor implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

using namespace clang::tooling;
using namespace llvm;

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");
static cl::opt<std::string> target_type("target-type",
                                        cl::desc("refactoring type name"),
                                        cl::value_desc("type name"),
                                        cl::ValueRequired, cl::NotHidden,
                                        cl::cat(MyToolCategory));

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...\n");

using namespace clang;
using namespace clang::ast_matchers;

class MemberFunctionCallMatcher : public MatchFinder::MatchCallback {
public:
  void run(const MatchFinder::MatchResult &Result) override {
    if (const CXXMemberCallExpr *MemberCall =
            Result.Nodes.getNodeAs<CXXMemberCallExpr>("memberCall")) {
      auto objExpr = MemberCall->getImplicitObjectArgument();
      auto endLoc = MemberCall->getExprLoc();

      auto exprRange = objExpr->getSourceRange();

      SourceLocation StartLoc = objExpr->getBeginLoc();

      const SourceManager &SM = *Result.SourceManager;
      const char *StartPtr = SM.getCharacterData(StartLoc);
      const char *EndPtr = SM.getCharacterData(endLoc);

      tooling::AtomicChange change(*Result.SourceManager,
                                   MemberCall->getExprLoc());
      const auto *ME = Result.Nodes.getNodeAs<MemberExpr>("member");
      size_t dropbackCount = ME->isArrow() ? 2 : 1;

      {
        auto length = EndPtr - StartPtr;
        objExprStrings.emplace_back(StartPtr, length);
        change.replace(*Result.SourceManager, StartLoc, EndPtr - StartPtr, "");
      }

      {
        // remove keyword template e.g. obj->template isa<T>
        auto legalObjStr = StringRef(objExprStrings.back()).rtrim();
        auto templateLoc = legalObjStr.find("template");
        if (templateLoc != std::string::npos)
          legalObjStr = legalObjStr.slice(0, templateLoc);

        // the obj is this when call the member function.
        if (legalObjStr.empty()) {
          objExprStrings.back() = "*this";
        } else {
          legalObjStr = legalObjStr.drop_back(dropbackCount);
          objExprStrings.back() =
              ME->isArrow() ? "*" + legalObjStr.str() : legalObjStr.str();
        }
        change.insert(*Result.SourceManager, MemberCall->getRParenLoc(),
                      objExprStrings.back());
      }
      changes.push_back(std::move(change));
    }
  }
  SmallVector<tooling::AtomicChange> changes;
  SmallVector<std::string> objExprStrings;
};

int main(int argc, const char **argv) {
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory,
                                                    cl::Optional, nullptr);
  if (!ExpectedParser) {
    // Fail gracefully for unsupported options.
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  MatchFinder Finder;

  auto MemberCallMatcher =
      cxxMemberCallExpr(
          callee(memberExpr().bind("member")),
          callee(cxxMethodDecl(
              ofClass(hasName(target_type)),
              hasAnyName("cast", "dyn_cast", "dyn_cast_or_null", "isa"))))
          .bind("memberCall");

  MemberFunctionCallMatcher memCallExpr;
  Finder.addMatcher(MemberCallMatcher, &memCallExpr);

  int ExitCode = Tool.run(newFrontendActionFactory(&Finder).get());
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  TextDiagnosticPrinter DiagnosticPrinter(errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);

  auto &FileMgr = Tool.getFiles();
  SourceManager Sources(Diagnostics, FileMgr);
  Rewriter rewriter(Sources, DefaultLangOptions);

  std::map<std::string, SmallVector<tooling::AtomicChange>> groupChanges;
  for (auto &change : memCallExpr.changes) {
    auto filePath = change.getFilePath();
    groupChanges[filePath].push_back(std::move(change));
  }

  auto applyOneChange = [](StringRef filePath,
                           ArrayRef<tooling::AtomicChange> changes) {
    tooling::ApplyChangesSpec Spec;
    Spec.Cleanup = false;

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferErr =
        llvm::MemoryBuffer::getFile(filePath);
    if (!BufferErr) {
      llvm::errs() << "error: failed to open " << filePath
                   << " for rewriting\n";
      return false;
    }
    auto Result = tooling::applyAtomicChanges(
        filePath, (*BufferErr)->getBuffer(), changes, Spec);
    if (!Result) {
      llvm::errs() << toString(Result.takeError());
      return false;
    }

    std::error_code EC;
    llvm::raw_fd_ostream OS(filePath, EC, llvm::sys::fs::OF_TextWithCRLF);
    if (EC) {
      llvm::errs() << EC.message() << "\n";
      return false;
    }
    OS << *Result;
    return true;
  };

  for (auto &change : groupChanges) {
    if (!applyOneChange(change.first, makeArrayRef(change.second))) {
      llvm::errs() << "apply file:" << change.first << " fail!";
    }
  }

  return ExitCode;
}