//===- unittests/Analysis/FlowSensitive/DeterminismTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport.h"
#include "clang/AST/Decl.h"
#include "clang/Analysis/FlowSensitive/AdornedCFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/NoopAnalysis.h"
#include "clang/Analysis/FlowSensitive/TypeErasedDataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/LLVM.h"
#include "clang/Testing/TestAST.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>

namespace clang::dataflow {

// Run a no-op analysis, and return a textual representation of the
// flow-condition at function exit.
std::string analyzeAndPrintExitCondition(llvm::StringRef Code) {
  DataflowAnalysisContext DACtx(std::make_unique<WatchedLiteralsSolver>());
  clang::TestAST AST(Code);
  const auto *Target =
      cast<FunctionDecl>(test::findValueDecl(AST.context(), "target"));
  Environment InitEnv(DACtx, *Target);
  auto ACFG = cantFail(AdornedCFG::build(*Target));

  NoopAnalysis Analysis(AST.context(), DataflowAnalysisOptions{});

  auto Result = runDataflowAnalysis(ACFG, Analysis, InitEnv);
  EXPECT_FALSE(!Result) << Result.takeError();

  Atom FinalFC = (*Result)[ACFG.getCFG().getExit().getBlockID()]
                     ->Env.getFlowConditionToken();
  std::string Textual;
  llvm::raw_string_ostream OS(Textual);
  DACtx.dumpFlowCondition(FinalFC, OS);
  return Textual;
}

TEST(DeterminismTest, NestedSwitch) {
  // Example extracted from real-world code that had wildly nondeterministic
  // analysis times.
  // Its flow condition depends on the order we join predecessor blocks.
  const char *Code = R"cpp(
    struct Tree;
    struct Rep {
      Tree *tree();
      int length;
    };
    struct Tree {
      int height();
      Rep *edge(int);
      int length;
    };
    struct RetVal {};
    int getInt();
    bool maybe();

    RetVal make(int size);
    inline RetVal target(int size, Tree& self) {
      Tree* tree = &self;
      const int height = self.height();
      Tree* n1 = tree;
      Tree* n2 = tree;
      switch (height) {
        case 3:
          tree = tree->edge(0)->tree();
          if (maybe()) return {};
          n2 = tree;
        case 2:
          tree = tree->edge(0)->tree();
          n1 = tree;
          if (maybe()) return {};
        case 1:
          tree = tree->edge(0)->tree();
          if (maybe()) return {};
        case 0:
          Rep* edge = tree->edge(0);
          if (maybe()) return {};
          int avail = getInt();
          if (avail == 0) return {};
          int delta = getInt();
          RetVal span = {};
          edge->length += delta;
          switch (height) {
            case 3:
              n1->length += delta;
            case 2:
              n1->length += delta;
            case 1:
              n1->length += delta;
            case 0:
              n1->length += delta;
              return span;
          }
          break;
      }
      return make(size);
    }
  )cpp";

  std::string Cond = analyzeAndPrintExitCondition(Code);
  for (unsigned I = 0; I < 10; ++I)
    EXPECT_EQ(Cond, analyzeAndPrintExitCondition(Code));
}

TEST(DeterminismTest, ValueMergeOrder) {
  // Artificial example whose final flow condition variable numbering depends
  // on the order in which we merge a, b, and c.
  const char *Code = R"cpp(
    bool target(bool a, bool b, bool c) {
      if (a)
        b = c;
      else
        c = b;
      return a && b && c;
    }
  )cpp";

  std::string Cond = analyzeAndPrintExitCondition(Code);
  for (unsigned I = 0; I < 10; ++I)
    EXPECT_EQ(Cond, analyzeAndPrintExitCondition(Code));
}

} // namespace clang::dataflow
