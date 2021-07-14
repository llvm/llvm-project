//===- unittests/AST/ASTPrint.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers to simplify testing of printing of AST constructs provided in the/
// form of the source code.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

namespace clang {

using PolicyAdjusterType =
    Optional<llvm::function_ref<void(PrintingPolicy &Policy)>>;

static void PrintStmt(raw_ostream &Out, const ASTContext *Context,
                      const Stmt *S, PolicyAdjusterType PolicyAdjuster) {
  assert(S != nullptr && "Expected non-null Stmt");
  PrintingPolicy Policy = Context->getPrintingPolicy();
  if (PolicyAdjuster)
    (*PolicyAdjuster)(Policy);
  S->printPretty(Out, /*Helper*/ nullptr, Policy);
}

class PrintMatch : public ast_matchers::MatchFinder::MatchCallback {
  SmallString<1024> Printed;
  unsigned NumFoundStmts;
  PolicyAdjusterType PolicyAdjuster;

public:
  PrintMatch(PolicyAdjusterType PolicyAdjuster)
      : NumFoundStmts(0), PolicyAdjuster(PolicyAdjuster) {}

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const Stmt *S = Result.Nodes.getNodeAs<Stmt>("id");
    if (!S)
      return;
    NumFoundStmts++;
    if (NumFoundStmts > 1)
      return;

    llvm::raw_svector_ostream Out(Printed);
    PrintStmt(Out, Result.Context, S, PolicyAdjuster);
  }

  StringRef getPrinted() const { return Printed; }

  unsigned getNumFoundStmts() const { return NumFoundStmts; }
};

template <typename T>
::testing::AssertionResult
PrintedStmtMatches(StringRef Code, const std::vector<std::string> &Args,
                   const T &NodeMatch, StringRef ExpectedPrinted,
                   PolicyAdjusterType PolicyAdjuster = None) {

  PrintMatch Printer(PolicyAdjuster);
  ast_matchers::MatchFinder Finder;
  Finder.addMatcher(NodeMatch, &Printer);
  std::unique_ptr<tooling::FrontendActionFactory> Factory(
      tooling::newFrontendActionFactory(&Finder));

  if (!tooling::runToolOnCodeWithArgs(Factory->create(), Code, Args))
    return testing::AssertionFailure()
           << "Parsing error in \"" << Code.str() << "\"";

  if (Printer.getNumFoundStmts() == 0)
    return testing::AssertionFailure() << "Matcher didn't find any statements";

  if (Printer.getNumFoundStmts() > 1)
    return testing::AssertionFailure()
           << "Matcher should match only one statement (found "
           << Printer.getNumFoundStmts() << ")";

  if (Printer.getPrinted() != ExpectedPrinted)
    return ::testing::AssertionFailure()
           << "Expected \"" << ExpectedPrinted.str() << "\", got \""
           << Printer.getPrinted().str() << "\"";

  return ::testing::AssertionSuccess();
}

} // namespace clang
