//===--- LRGraphTest.cpp - LRGraph tests -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Pseudo/LRGraph.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang {
namespace syntax {
namespace pseudo {
namespace {

TEST(LRGraph, Build) {
  struct TestCase {
    llvm::StringRef BNF;
    llvm::StringRef ExpectedStates;
  };

  TestCase Cases[] = {{
                          R"bnf(
_ := expr
expr := IDENTIFIER
      )bnf",
                          R"(States:
State 0
    _ :=  • expr
    expr :=  • IDENTIFIER
State 1
    _ := expr • 
State 2
    expr := IDENTIFIER • 
0 ->[expr] 1
0 ->[IDENTIFIER] 2
)"},
                      {// A grammar with a S/R conflict in SLR table:
                       // (id-id)-id, or id-(id-id).
                       R"bnf(
_ := expr
expr := expr - expr  # S/R conflict at state 4 on '-' token
expr := IDENTIFIER
      )bnf",
                       R"(States:
State 0
    _ :=  • expr
    expr :=  • expr - expr
    expr :=  • IDENTIFIER
State 1
    _ := expr • 
    expr := expr • - expr
State 2
    expr := IDENTIFIER • 
State 3
    expr :=  • expr - expr
    expr := expr - • expr
    expr :=  • IDENTIFIER
State 4
    expr := expr - expr • 
    expr := expr • - expr
0 ->[expr] 1
0 ->[IDENTIFIER] 2
1 ->[-] 3
3 ->[expr] 4
3 ->[IDENTIFIER] 2
4 ->[-] 3
)"}};
  for (const auto &C : Cases) {
    std::vector<std::string> Diags;
    auto G = Grammar::parseBNF(C.BNF, Diags);
    ASSERT_THAT(Diags, testing::IsEmpty());
    auto LR0 = LRGraph::buildLR0(*G);
    EXPECT_EQ(LR0.dumpForTests(*G), C.ExpectedStates);
  }
}

} // namespace
} // namespace pseudo
} // namespace syntax
} // namespace clang
