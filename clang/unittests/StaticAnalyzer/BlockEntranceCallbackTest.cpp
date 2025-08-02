//===- unittests/StaticAnalyzer/BlockEntranceCallbackTest.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/ProgramPoint.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ento;

namespace {

class BlockEntranceCallbackTester final : public Checker<check::BlockEntrance> {
  const BugType Bug{this, "BlockEntranceTester"};

public:
  void checkBlockEntrance(const BlockEntrance &Entrance,
                          CheckerContext &C) const {
    ExplodedNode *Node = C.generateNonFatalErrorNode(C.getState());
    if (!Node)
      return;

    const auto *FD =
        cast<FunctionDecl>(C.getLocationContext()->getStackFrame()->getDecl());

    std::string Description = llvm::formatv(
        "Within '{0}' B{1} -> B{2}", FD->getIdentifier()->getName(),
        Entrance.getPreviousBlock()->getBlockID(),
        Entrance.getBlock()->getBlockID());
    auto Report =
        std::make_unique<PathSensitiveBugReport>(Bug, Description, Node);
    C.emitReport(std::move(Report));
  }
};

class BranchConditionCallbackTester final
    : public Checker<check::BranchCondition> {
  const BugType Bug{this, "BranchConditionCallbackTester"};

public:
  void checkBranchCondition(const Stmt *Condition, CheckerContext &C) const {
    ExplodedNode *Node = C.generateNonFatalErrorNode(C.getState());
    if (!Node)
      return;
    const auto *FD =
        cast<FunctionDecl>(C.getLocationContext()->getStackFrame()->getDecl());

    std::string Buffer =
        (llvm::Twine("Within '") + FD->getIdentifier()->getName() +
         "': branch condition '")
            .str();
    llvm::raw_string_ostream OS(Buffer);
    Condition->printPretty(OS, /*Helper=*/nullptr,
                           C.getASTContext().getPrintingPolicy());
    OS << "'";
    auto Report = std::make_unique<PathSensitiveBugReport>(Bug, Buffer, Node);
    C.emitReport(std::move(Report));

    C.addTransition();
  }
};

template <typename Checker> void registerChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<Checker>();
}

bool shouldAlwaysRegister(const CheckerManager &) { return true; }

void addBlockEntranceTester(AnalysisASTConsumer &AnalysisConsumer,
                            AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages.emplace_back("test.BlockEntranceTester", true);
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker(&registerChecker<BlockEntranceCallbackTester>,
                        &shouldAlwaysRegister, "test.BlockEntranceTester",
                        "EmptyDescription");
  });
}

void addBranchConditionTester(AnalysisASTConsumer &AnalysisConsumer,
                              AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages.emplace_back("test.BranchConditionTester", true);
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker(&registerChecker<BranchConditionCallbackTester>,
                        &shouldAlwaysRegister, "test.BranchConditionTester",
                        "EmptyDescription");
  });
}

llvm::SmallVector<StringRef> parseEachDiag(StringRef Diags) {
  llvm::SmallVector<StringRef> Fragments;
  llvm::SplitString(Diags, Fragments, "\n");
  // Drop the prefix like "test.BlockEntranceTester: " from each fragment.
  for (StringRef &Fragment : Fragments) {
    Fragment = Fragment.drop_until([](char Ch) { return Ch == ' '; });
    Fragment.consume_front(" ");
  }
  llvm::sort(Fragments);
  return Fragments;
}

template <AddCheckerFn Fn = addBlockEntranceTester, AddCheckerFn... Fns>
bool runChecker(const std::string &Code, std::string &Diags) {
  std::string RawDiags;
  bool Res = runCheckerOnCode<Fn, Fns...>(Code, RawDiags,
                                          /*OnlyEmitWarnings=*/true);
  llvm::raw_string_ostream OS(Diags);
  llvm::interleave(parseEachDiag(RawDiags), OS, "\n");
  return Res;
}

[[maybe_unused]] void dumpCFGAndEgraph(AnalysisASTConsumer &AnalysisConsumer,
                                       AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages.emplace_back("debug.DumpCFG", true);
  AnOpts.CheckersAndPackages.emplace_back("debug.ViewExplodedGraph", true);
}

/// Use this instead of \c runChecker to enable the debugging a test case.
template <AddCheckerFn... Fns>
[[maybe_unused]] bool debugChecker(const std::string &Code,
                                   std::string &Diags) {
  return runChecker<dumpCFGAndEgraph, Fns...>(Code, Diags);
}

std::string expected(SmallVector<StringRef> Diags) {
  llvm::sort(Diags);
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  llvm::interleave(Diags, OS, "\n");
  return Result;
}

TEST(BlockEntranceTester, FromEntryToExit) {
  constexpr auto Code = R"cpp(
  void top() {
    // empty
  })cpp";

  std::string Diags;
  // Use "debugChecker" instead of "runChecker" for debugging.
  EXPECT_TRUE(runChecker(Code, Diags));
  EXPECT_EQ(expected({"Within 'top' B1 -> B0"}), Diags);
}

TEST(BlockEntranceTester, SingleOpaqueIfCondition) {
  constexpr auto Code = R"cpp(
  bool coin();
  int glob;
  void top() {
    if (coin()) {
      glob = 1;
    } else {
      glob = 2;
    }
    glob = 3;
  })cpp";

  std::string Diags;
  // Use "debugChecker" instead of "runChecker" for debugging.
  EXPECT_TRUE(runChecker(Code, Diags));
  EXPECT_EQ(expected({
                "Within 'top' B1 -> B0",
                "Within 'top' B2 -> B1",
                "Within 'top' B3 -> B1",
                "Within 'top' B4 -> B2",
                "Within 'top' B4 -> B3",
                "Within 'top' B5 -> B4",
            }),
            Diags);
  // entry true                   exit
  //  B5 -------> B4 --> B2 --> B1 --> B0
  //  |                         ^
  //  | false                   |
  //  v                         |
  //  B3 -----------------------+
}

TEST(BlockEntranceTester, TrivialIfCondition) {
  constexpr auto Code = R"cpp(
  bool coin();
  int glob;
  void top() {
    int cond = true;
    if (cond) {
      glob = 1;
    } else {
      glob = 2;
    }
    glob = 3;
  })cpp";

  std::string Diags;
  // Use "debugChecker" instead of "runChecker" for debugging.
  EXPECT_TRUE(runChecker(Code, Diags));
  EXPECT_EQ(expected({
                "Within 'top' B1 -> B0",
                "Within 'top' B3 -> B1",
                "Within 'top' B4 -> B3",
                "Within 'top' B5 -> B4",
            }),
            Diags);
  // entry  true                         exit
  // B5 ----------> B4 --> B3 --> B1 --> B0
}

TEST(BlockEntranceTester, AcrossFunctions) {
  constexpr auto Code = R"cpp(
  bool coin();
  int glob;
  void nested() { glob = 1; }
  void top() {
    glob = 0;
    nested();
    glob = 2;
  })cpp";

  std::string Diags;
  // Use "debugChecker" instead of "runChecker" for debugging.
  EXPECT_TRUE(runChecker(Code, Diags));
  EXPECT_EQ(
      expected({
          // Going from the "top" entry artificial node to the "top" body.
          // Ideally, we shouldn't observe this edge because it's artificial.
          "Within 'top' B2 -> B1",

          // We encounter the call to "nested()" in the "top" body, thus we have
          // a "CallEnter" node, but importantly, we also elide the transition
          // to the "entry" node of "nested()".
          // We only see the edge from the "nested()" entry to the "nested()"
          // body:
          "Within 'nested' B2 -> B1",

          // Once we return from "nested()", we transition to the "exit" node of
          // "nested()":
          "Within 'nested' B1 -> B0",

          // We will eventually return to the "top" body, thus we transition to
          // its "exit" node:
          "Within 'top' B1 -> B0",
      }),
      Diags);
}

TEST(BlockEntranceTester, ShortCircuitingLogicalOperator) {
  constexpr auto Code = R"cpp(
  bool coin();
  void top(int x) {
    int v = 0;
    if (coin() && (v = x)) {
      v = 2;
    }
    v = 3;
  })cpp";
  //                        coin(): false
  //              +--------------------------------+
  // entry        |                                v         exit
  // +----+     +----+     +----+     +----+     +----+     +----+
  // | B5 | --> | B4 | --> | B3 | --> | B2 | --> | B1 | --> | B0 |
  // +----+     +----+     +----+     +----+     +----+     +----+
  //                         |                     ^
  //                         +---------------------+
  //                            (v = x): false

  std::string Diags;
  // Use "debugChecker" instead of "runChecker" for debugging.
  EXPECT_TRUE(runChecker(Code, Diags));
  EXPECT_EQ(expected({
                "Within 'top' B1 -> B0",
                "Within 'top' B2 -> B1",
                "Within 'top' B3 -> B1",
                "Within 'top' B3 -> B2",
                "Within 'top' B4 -> B1",
                "Within 'top' B4 -> B3",
                "Within 'top' B5 -> B4",
            }),
            Diags);
}

TEST(BlockEntranceTester, Switch) {
  constexpr auto Code = R"cpp(
  bool coin();
  int top(int x) {
    int v = 0;
    switch (x) {
      case 1:  v = 10; break;
      case 2:  v = 20; break;
      default: v = 30; break;
    }
    return v;
  })cpp";
  //            +----+
  //            | B5 | -------------------------+
  //            +----+                          |
  //              ^ [case 1]                    |
  // entry        |                             v         exit
  // +----+     +----+  [default]  +----+     +----+     +----+
  // | B6 | --> | B2 | ----------> | B3 | --> | B1 | --> | B0 |
  // +----+     +----+             +----+     +----+     +----+
  //              |                             ^
  //              v [case 2]                    |
  //            +----+                          |
  //            | B4 | -------------------------+
  //            +----+

  std::string Diags;
  // Use "debugChecker" instead of "runChecker" for debugging.
  EXPECT_TRUE(runChecker(Code, Diags));
  EXPECT_EQ(expected({
                "Within 'top' B1 -> B0",
                "Within 'top' B2 -> B3",
                "Within 'top' B2 -> B4",
                "Within 'top' B2 -> B5",
                "Within 'top' B3 -> B1",
                "Within 'top' B4 -> B1",
                "Within 'top' B5 -> B1",
                "Within 'top' B6 -> B2",
            }),
            Diags);
}

TEST(BlockEntranceTester, BlockEntranceVSBranchCondition) {
  constexpr auto Code = R"cpp(
  bool coin();
  int top(int x) {
    int v = 0;
    switch (x) {
      default: v = 30; break;
    }
    if (x == 6) {
      v = 40;
    }
    return v;
  })cpp";
  std::string Diags;
  // Use "debugChecker" instead of "runChecker" for debugging.
  EXPECT_TRUE((runChecker<addBlockEntranceTester, addBranchConditionTester>(
      Code, Diags)));
  EXPECT_EQ(expected({
                "Within 'top' B1 -> B0",
                "Within 'top' B2 -> B1",
                "Within 'top' B3 -> B1",
                "Within 'top' B3 -> B2",
                "Within 'top' B4 -> B5",
                "Within 'top' B5 -> B3",
                "Within 'top' B6 -> B4",
                "Within 'top': branch condition 'x == 6'",
            }),
            Diags);
}

} // namespace
