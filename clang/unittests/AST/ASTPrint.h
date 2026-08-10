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

using PrintingPolicyAdjuster = llvm::function_ref<void(PrintingPolicy &Policy)>;

template <typename NodeType>
using NodePrinter =
    std::function<void(llvm::raw_ostream &Out, const ASTContext *Context,
                       const NodeType *Node,
                       PrintingPolicyAdjuster PolicyAdjuster)>;

template <typename NodeType>
using NodeFilter = std::function<bool(const NodeType *Node)>;

template <typename NodeType>
class PrintMatch : public ast_matchers::MatchFinder::MatchCallback {
  using PrinterT = NodePrinter<NodeType>;
  using FilterT = NodeFilter<NodeType>;

  SmallString<1024> Printed;
  unsigned NumFoundNodes;
  PrinterT Printer;
  FilterT Filter;
  PrintingPolicyAdjuster PolicyAdjuster;

public:
  PrintMatch(PrinterT Printer, PrintingPolicyAdjuster PolicyAdjuster,
             FilterT Filter)
      : NumFoundNodes(0), Printer(std::move(Printer)),
        Filter(std::move(Filter)), PolicyAdjuster(PolicyAdjuster) {}

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const NodeType *N = Result.Nodes.getNodeAs<NodeType>("id");
    if (!N || !Filter(N))
      return;
    NumFoundNodes++;
    if (NumFoundNodes > 1)
      return;

    llvm::raw_svector_ostream Out(Printed);
    Printer(Out, Result.Context, N, PolicyAdjuster);
  }

  StringRef getPrinted() const { return Printed; }

  unsigned getNumFoundNodes() const { return NumFoundNodes; }
};

template <typename NodeType> bool NoNodeFilter(const NodeType *) {
  return true;
}

template <typename NodeType, typename Matcher>
::testing::AssertionResult
PrintedNodeMatches(StringRef Code, const std::vector<std::string> &Args,
                   const Matcher &NodeMatch, StringRef ExpectedPrinted,
                   StringRef FileName, NodePrinter<NodeType> Printer,
                   PrintingPolicyAdjuster PolicyAdjuster = nullptr,
                   bool AllowError = false,
                   // Would like to use a lambda for the default value, but that
                   // trips gcc 7 up.
                   NodeFilter<NodeType> Filter = &NoNodeFilter<NodeType>) {

  PrintMatch<NodeType> Callback(Printer, PolicyAdjuster, Filter);
  ast_matchers::MatchFinder Finder;
  Finder.addMatcher(NodeMatch, &Callback);
  std::unique_ptr<tooling::FrontendActionFactory> Factory(
      tooling::newFrontendActionFactory(&Finder));

  bool ToolResult;
  if (FileName.empty()) {
    ToolResult = tooling::runToolOnCodeWithArgs(Factory->create(), Code, Args);
  } else {
    ToolResult =
        tooling::runToolOnCodeWithArgs(Factory->create(), Code, Args, FileName);
  }
  if (!ToolResult && !AllowError)
    return testing::AssertionFailure()
           << "Parsing error in \"" << Code.str() << "\"";

  if (Callback.getNumFoundNodes() == 0)
    return testing::AssertionFailure() << "Matcher didn't find any nodes";

  if (Callback.getNumFoundNodes() > 1)
    return testing::AssertionFailure()
           << "Matcher should match only one node (found "
           << Callback.getNumFoundNodes() << ")";

  if (Callback.getPrinted() != ExpectedPrinted)
    return ::testing::AssertionFailure()
           << "Expected \"" << ExpectedPrinted.str() << "\", got \""
           << Callback.getPrinted().str() << "\"";

  return ::testing::AssertionSuccess();
}

} // namespace clang
