//===--- DisambiguateTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Disambiguate.h"
#include "clang-pseudo/Forest.h"
#include "clang-pseudo/Token.h"
#include "clang/Basic/TokenKinds.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

namespace clang {
namespace pseudo {
namespace {
using testing::ElementsAre;
using testing::Pair;
using testing::UnorderedElementsAre;

// Common disambiguation test fixture.
// This is the ambiguous forest representing parses of 'a * b;'.
class DisambiguateTest : public ::testing::Test {
protected:
  // Greatly simplified C++ grammar.
  enum Symbol : SymbolID {
    Statement,
    Declarator,
    Expression,
    DeclSpecifier,
    Type,
    Template,
  };
  enum Rule : RuleID {
    /* LHS__RHS1_RHS2 means LHS := RHS1 RHS2 */
    Statement__DeclSpecifier_Declarator_Semi,
    Declarator__Star_Declarator,
    Declarator__Identifier,
    Statement__Expression_Semi,
    Expression__Expression_Star_Expression,
    Expression__Identifier,
    DeclSpecifier__Type,
    DeclSpecifier__Template,
    Type__Identifier,
    Template__Identifier,
  };

  ForestArena Arena;
  ForestNode &A = Arena.createTerminal(tok::identifier, 0);
  ForestNode &Star = Arena.createTerminal(tok::star, 1);
  ForestNode &B = Arena.createTerminal(tok::identifier, 2);
  ForestNode &Semi = Arena.createTerminal(tok::semi, 3);

  // Parse as multiplication expression.
  ForestNode &AExpr =
      Arena.createSequence(Expression, Expression__Identifier, &A);
  ForestNode &BExpr =
      Arena.createSequence(Expression, Expression__Identifier, &B);
  ForestNode &Expr =
      Arena.createSequence(Expression, Expression__Expression_Star_Expression,
                           {&AExpr, &Star, &BExpr});
  ForestNode &ExprStmt = Arena.createSequence(
      Statement, Statement__Expression_Semi, {&Expr, &Semi});
  // Parse as declaration (`a` may be CTAD or not).
  ForestNode &AType =
      Arena.createSequence(DeclSpecifier, DeclSpecifier__Type,
                           &Arena.createSequence(Type, Type__Identifier, &A));
  ForestNode &ATemplate = Arena.createSequence(
      DeclSpecifier, DeclSpecifier__Template,
      &Arena.createSequence(Template, Template__Identifier, &A));
  ForestNode &DeclSpec =
      Arena.createAmbiguous(DeclSpecifier, {&AType, &ATemplate});
  ForestNode &BDeclarator =
      Arena.createSequence(Declarator, Declarator__Identifier, &B);
  ForestNode &BPtr = Arena.createSequence(
      Declarator, Declarator__Star_Declarator, {&Star, &BDeclarator});
  ForestNode &DeclStmt =
      Arena.createSequence(Statement, Statement__DeclSpecifier_Declarator_Semi,
                           {&DeclSpec, &Star, &BDeclarator});
  // Top-level ambiguity
  ForestNode &Stmt = Arena.createAmbiguous(Statement, {&ExprStmt, &DeclStmt});
};

TEST_F(DisambiguateTest, Remove) {
  Disambiguation D;
  D.try_emplace(&Stmt, 1);     // statement is a declaration, not an expression
  D.try_emplace(&DeclSpec, 0); // a is a type, not a (CTAD) template
  ForestNode *Root = &Stmt;
  removeAmbiguities(Root, D);

  EXPECT_EQ(Root, &DeclStmt);
  EXPECT_THAT(DeclStmt.elements(), ElementsAre(&AType, &Star, &BDeclarator));
}

TEST_F(DisambiguateTest, DummyStrategy) {
  Disambiguation D = disambiguate(&Stmt, {});
  EXPECT_THAT(D, UnorderedElementsAre(Pair(&Stmt, 1), Pair(&DeclSpec, 1)));

  ForestNode *Root = &Stmt;
  removeAmbiguities(Root, D);
  EXPECT_EQ(Root, &DeclStmt);
  EXPECT_THAT(DeclStmt.elements(),
              ElementsAre(&ATemplate, &Star, &BDeclarator));
}

} // namespace
} // namespace pseudo
} // namespace clang
