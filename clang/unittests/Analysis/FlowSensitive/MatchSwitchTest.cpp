//===- unittests/Analysis/FlowSensitive/MatchSwitchTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

using namespace clang;
using namespace dataflow;
using namespace ast_matchers;

namespace {

TEST(MatchSwitchTest, Stmts) {
  std::string Code = R"(
    void Foo();
    void Bar();
    void f() {
      int X = 1;
      Foo();
      Bar();
    }
  )";
  auto Unit = tooling::buildASTFromCode(Code);
  auto &Ctx = Unit->getASTContext();

  llvm::StringRef XStr = "X";
  llvm::StringRef FooStr = "Foo";
  llvm::StringRef BarStr = "Bar";

  auto XMatcher = declStmt(hasSingleDecl(varDecl(hasName(XStr))));
  auto FooMatcher = callExpr(callee(functionDecl(hasName(FooStr))));
  auto BarMatcher = callExpr(callee(functionDecl(hasName(BarStr))));

  ASTMatchSwitch<Stmt, llvm::StringRef> MS =
      ASTMatchSwitchBuilder<Stmt, llvm::StringRef>()
          .CaseOf<Stmt>(XMatcher,
                        [&XStr](const Stmt *, const MatchFinder::MatchResult &,
                                llvm::StringRef &State) { State = XStr; })
          .CaseOf<Stmt>(FooMatcher,
                        [&FooStr](const Stmt *,
                                  const MatchFinder::MatchResult &,
                                  llvm::StringRef &State) { State = FooStr; })
          .Build();
  llvm::StringRef State;

  // State modified from the first case of the switch
  const auto *X = selectFirst<Stmt>(XStr, match(XMatcher.bind(XStr), Ctx));
  MS(*X, Ctx, State);
  EXPECT_EQ(State, XStr);

  // State modified from the second case of the switch
  const auto *Foo =
      selectFirst<Stmt>(FooStr, match(FooMatcher.bind(FooStr), Ctx));
  MS(*Foo, Ctx, State);
  EXPECT_EQ(State, FooStr);

  // State unmodified, no case defined for calling Bar
  const auto *Bar =
      selectFirst<Stmt>(BarStr, match(BarMatcher.bind(BarStr), Ctx));
  MS(*Bar, Ctx, State);
  EXPECT_EQ(State, FooStr);
}

TEST(MatchSwitchTest, CtorInitializers) {
  std::string Code = R"(
    struct f {
      int i;
      int j;
      int z;
      f(): i(1), j(1), z(1) {}
    };
  )";
  auto Unit = tooling::buildASTFromCode(Code);
  auto &Ctx = Unit->getASTContext();

  llvm::StringRef IStr = "i";
  llvm::StringRef JStr = "j";
  llvm::StringRef ZStr = "z";

  auto InitI = cxxCtorInitializer(forField(hasName(IStr)));
  auto InitJ = cxxCtorInitializer(forField(hasName(JStr)));
  auto InitZ = cxxCtorInitializer(forField(hasName(ZStr)));

  ASTMatchSwitch<CXXCtorInitializer, llvm::StringRef> MS =
      ASTMatchSwitchBuilder<CXXCtorInitializer, llvm::StringRef>()
          .CaseOf<CXXCtorInitializer>(
              InitI, [&IStr](const CXXCtorInitializer *,
                             const MatchFinder::MatchResult &,
                             llvm::StringRef &State) { State = IStr; })
          .CaseOf<CXXCtorInitializer>(
              InitJ, [&JStr](const CXXCtorInitializer *,
                             const MatchFinder::MatchResult &,
                             llvm::StringRef &State) { State = JStr; })
          .Build();
  llvm::StringRef State;

  // State modified from the first case of the switch
  const auto *I =
      selectFirst<CXXCtorInitializer>(IStr, match(InitI.bind(IStr), Ctx));
  MS(*I, Ctx, State);
  EXPECT_EQ(State, IStr);

  // State modified from the second case of the switch
  const auto *J =
      selectFirst<CXXCtorInitializer>(JStr, match(InitJ.bind(JStr), Ctx));
  MS(*J, Ctx, State);
  EXPECT_EQ(State, JStr);

  // State unmodified, no case defined for the initializer of z
  const auto *Z =
      selectFirst<CXXCtorInitializer>(ZStr, match(InitZ.bind(ZStr), Ctx));
  MS(*Z, Ctx, State);
  EXPECT_EQ(State, JStr);
}

TEST(MatchSwitchTest, ReturnNonVoid) {
  auto Unit =
      tooling::buildASTFromCode("void f() { int x = 42; }", "input.cc",
                                std::make_shared<PCHContainerOperations>());
  auto &Context = Unit->getASTContext();
  const auto *S =
      selectFirst<FunctionDecl>(
          "f",
          match(functionDecl(isDefinition(), hasName("f")).bind("f"), Context))
          ->getBody();

  ASTMatchSwitch<Stmt, const int, std::vector<int>> Switch =
      ASTMatchSwitchBuilder<Stmt, const int, std::vector<int>>()
          .CaseOf<Stmt>(stmt(),
                        [](const Stmt *, const MatchFinder::MatchResult &,
                           const int &State) -> std::vector<int> {
                          return {1, State, 3};
                        })
          .Build();
  std::vector<int> Actual = Switch(*S, Context, 7);
  std::vector<int> Expected{1, 7, 3};
  EXPECT_EQ(Actual, Expected);
}

} // namespace
