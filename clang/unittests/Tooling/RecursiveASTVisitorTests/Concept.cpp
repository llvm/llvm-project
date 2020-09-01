//===- unittest/Tooling/RecursiveASTVisitorTests/Concept.cpp----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/AST/ExprConcepts.h"

using namespace clang;

namespace {

struct ConceptVisitor : ExpectedLocationVisitor<ConceptVisitor> {
  bool VisitConceptSpecializationExpr(ConceptSpecializationExpr *E) {
    ++ConceptSpecializationExprsVisited;
    return true;
  }
  bool TraverseConceptReference(const ConceptReference &R) {
    ++ConceptReferencesTraversed;
    return true;
  }

  int ConceptSpecializationExprsVisited = 0;
  int ConceptReferencesTraversed = 0;
};

TEST(RecursiveASTVisitor, ConstrainedParameter) {
  ConceptVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver("template <typename T> concept Fooable = true;\n"
                              "template <Fooable T> void bar(T);",
                              ConceptVisitor::Lang_CXX2a));
  // Check that we visit the "Fooable T" template parameter's TypeConstraint's
  // ImmediatelyDeclaredConstraint, which is a ConceptSpecializationExpr.
  EXPECT_EQ(1, Visitor.ConceptSpecializationExprsVisited);
  // There are two ConceptReference objects in the AST: the base subobject
  // of the ConceptSpecializationExpr, and the base subobject of the
  // TypeConstraint itself. To avoid traversing the concept and arguments
  // multiple times, we only traverse one.
  EXPECT_EQ(1, Visitor.ConceptReferencesTraversed);
}

} // end anonymous namespace
