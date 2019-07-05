//===- unittests/Analysis/CFGDominatorTree.cpp - CFG tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CFGBuildResult.h"
#include "clang/Analysis/Analyses/Dominators.h"
#include "gtest/gtest.h"

namespace clang {
namespace analysis {
namespace {

TEST(CFGDominatorTree, DomTree) {
  const char *Code = R"(enum Kind {
                          A
                        };

                        void f() {
                          switch(Kind{}) {
                          case A:
                            break;
                          }
                        })";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());

  //  [B3 (ENTRY)]  -> [B1] -> [B2] -> [B0 (EXIT)]
  //                  switch  case A

  CFG *cfg = Result.getCFG();

  // Sanity checks.
  EXPECT_EQ(cfg->size(), 4u);

  CFGBlock *ExitBlock = *cfg->begin();
  EXPECT_EQ(ExitBlock, &cfg->getExit());

  CFGBlock *SwitchBlock = *(cfg->begin() + 1);

  CFGBlock *CaseABlock = *(cfg->begin() + 2);

  CFGBlock *EntryBlock = *(cfg->begin() + 3);
  EXPECT_EQ(EntryBlock, &cfg->getEntry());

  // Test the dominator tree.
  CFGDomTree Dom;
  Dom.buildDominatorTree(cfg);

  EXPECT_TRUE(Dom.dominates(ExitBlock, ExitBlock));
  EXPECT_FALSE(Dom.properlyDominates(ExitBlock, ExitBlock));
  EXPECT_TRUE(Dom.dominates(CaseABlock, ExitBlock));
  EXPECT_TRUE(Dom.dominates(SwitchBlock, ExitBlock));
  EXPECT_TRUE(Dom.dominates(EntryBlock, ExitBlock));

  EXPECT_TRUE(Dom.dominates(CaseABlock, CaseABlock));
  EXPECT_FALSE(Dom.properlyDominates(CaseABlock, CaseABlock));
  EXPECT_TRUE(Dom.dominates(SwitchBlock, CaseABlock));
  EXPECT_TRUE(Dom.dominates(EntryBlock, CaseABlock));

  EXPECT_TRUE(Dom.dominates(SwitchBlock, SwitchBlock));
  EXPECT_FALSE(Dom.properlyDominates(SwitchBlock, SwitchBlock));
  EXPECT_TRUE(Dom.dominates(EntryBlock, SwitchBlock));

  EXPECT_TRUE(Dom.dominates(EntryBlock, EntryBlock));
  EXPECT_FALSE(Dom.properlyDominates(EntryBlock, EntryBlock));

  // Test the post dominator tree.

  CFGPostDomTree PostDom;
  PostDom.buildDominatorTree(cfg);

  EXPECT_TRUE(PostDom.dominates(ExitBlock, EntryBlock));
  EXPECT_TRUE(PostDom.dominates(CaseABlock, EntryBlock));
  EXPECT_TRUE(PostDom.dominates(SwitchBlock, EntryBlock));
  EXPECT_TRUE(PostDom.dominates(EntryBlock, EntryBlock));
  EXPECT_FALSE(Dom.properlyDominates(EntryBlock, EntryBlock));

  EXPECT_TRUE(PostDom.dominates(ExitBlock, SwitchBlock));
  EXPECT_TRUE(PostDom.dominates(CaseABlock, SwitchBlock));
  EXPECT_TRUE(PostDom.dominates(SwitchBlock, SwitchBlock));
  EXPECT_FALSE(Dom.properlyDominates(SwitchBlock, SwitchBlock));

  EXPECT_TRUE(PostDom.dominates(ExitBlock, CaseABlock));
  EXPECT_TRUE(PostDom.dominates(CaseABlock, CaseABlock));
  EXPECT_FALSE(Dom.properlyDominates(CaseABlock, CaseABlock));

  EXPECT_TRUE(PostDom.dominates(ExitBlock, ExitBlock));
  EXPECT_FALSE(Dom.properlyDominates(ExitBlock, ExitBlock));

  // Tests for the post dominator tree's virtual root.
  EXPECT_TRUE(PostDom.dominates(nullptr, EntryBlock));
  EXPECT_TRUE(PostDom.dominates(nullptr, SwitchBlock));
  EXPECT_TRUE(PostDom.dominates(nullptr, CaseABlock));
  EXPECT_TRUE(PostDom.dominates(nullptr, ExitBlock));
}

} // namespace
} // namespace analysis
} // namespace clang
