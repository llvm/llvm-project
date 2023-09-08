//===- unittest/Tooling/RecursiveASTVisitorTests/Concept.cpp----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/AST/ASTConcept.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/Type.h"

using namespace clang;

namespace {

struct ConceptVisitor : ExpectedLocationVisitor<ConceptVisitor> {
  bool VisitConceptSpecializationExpr(ConceptSpecializationExpr *E) {
    ++ConceptSpecializationExprsVisited;
    return true;
  }
  bool TraverseTypeConstraint(const TypeConstraint *C) {
    ++TypeConstraintsTraversed;
    return ExpectedLocationVisitor::TraverseTypeConstraint(C);
  }
  bool TraverseConceptRequirement(concepts::Requirement *R) {
    ++ConceptRequirementsTraversed;
    return ExpectedLocationVisitor::TraverseConceptRequirement(R);
  }
  bool TraverseConceptReference(ConceptReference *CR) {
    ++ConceptReferencesTraversed;
    return ExpectedLocationVisitor::TraverseConceptReference(CR);
  }
  bool VisitConceptReference(ConceptReference *CR) {
    ++ConceptReferencesVisited;
    return true;
  }

  bool shouldVisitImplicitCode() { return ShouldVisitImplicitCode; }

  int ConceptSpecializationExprsVisited = 0;
  int TypeConstraintsTraversed = 0;
  int ConceptRequirementsTraversed = 0;
  int ConceptReferencesTraversed = 0;
  int ConceptReferencesVisited = 0;
  bool ShouldVisitImplicitCode = false;
};

TEST(RecursiveASTVisitor, Concepts) {
  ConceptVisitor Visitor;
  Visitor.ShouldVisitImplicitCode = true;
  EXPECT_TRUE(Visitor.runOver("template <typename T> concept Fooable = true;\n"
                              "template <Fooable T> void bar(T);",
                              ConceptVisitor::Lang_CXX2a));
  // Check that we traverse the "Fooable T" template parameter's
  // TypeConstraint's ImmediatelyDeclaredConstraint, which is a
  // ConceptSpecializationExpr.
  EXPECT_EQ(1, Visitor.ConceptSpecializationExprsVisited);
  // Also check we traversed the TypeConstraint that produced the expr.
  EXPECT_EQ(1, Visitor.TypeConstraintsTraversed);
  EXPECT_EQ(1, Visitor.ConceptReferencesTraversed);
  EXPECT_EQ(1, Visitor.ConceptReferencesVisited);

  Visitor = {}; // Don't visit implicit code now.
  EXPECT_TRUE(Visitor.runOver("template <typename T> concept Fooable = true;\n"
                              "template <Fooable T> void bar(T);",
                              ConceptVisitor::Lang_CXX2a));
  // Check that we only visit the TypeConstraint, but not the implicitly
  // generated immediately declared expression.
  EXPECT_EQ(0, Visitor.ConceptSpecializationExprsVisited);
  EXPECT_EQ(1, Visitor.TypeConstraintsTraversed);
  EXPECT_EQ(1, Visitor.ConceptReferencesTraversed);
  EXPECT_EQ(1, Visitor.ConceptReferencesVisited);

  Visitor = {};
  EXPECT_TRUE(Visitor.runOver("template <class T> concept A = true;\n"
                              "template <class T> struct vector {};\n"
                              "template <class T> concept B = requires(T x) {\n"
                              "  typename vector<T*>;\n"
                              "  {x} -> A;\n"
                              "  requires true;\n"
                              "};",
                              ConceptVisitor::Lang_CXX2a));
  EXPECT_EQ(3, Visitor.ConceptRequirementsTraversed);
  EXPECT_EQ(1, Visitor.ConceptReferencesTraversed);
  EXPECT_EQ(1, Visitor.ConceptReferencesVisited);
}

struct VisitDeclOnlyOnce : ExpectedLocationVisitor<VisitDeclOnlyOnce> {
  bool VisitConceptDecl(ConceptDecl *D) {
    ++ConceptDeclsVisited;
    return true;
  }

  bool VisitAutoType(AutoType *) {
    ++AutoTypeVisited;
    return true;
  }
  bool VisitAutoTypeLoc(AutoTypeLoc) {
    ++AutoTypeLocVisited;
    return true;
  }
  bool VisitConceptReference(ConceptReference *) {
    ++ConceptReferencesVisited;
    return true;
  }

  bool TraverseVarDecl(VarDecl *V) {
    // The base traversal visits only the `TypeLoc`.
    // However, in the test we also validate the underlying `QualType`.
    TraverseType(V->getType());
    return ExpectedLocationVisitor::TraverseVarDecl(V);
  }

  bool shouldWalkTypesOfTypeLocs() { return false; }

  int ConceptDeclsVisited = 0;
  int AutoTypeVisited = 0;
  int AutoTypeLocVisited = 0;
  int ConceptReferencesVisited = 0;
};

TEST(RecursiveASTVisitor, ConceptDeclInAutoType) {
  // Check `AutoType` and `AutoTypeLoc` do not repeatedly traverse the
  // underlying concept.
  VisitDeclOnlyOnce Visitor;
  Visitor.runOver("template <class T> concept A = true;\n"
                  "A auto i = 0;\n",
                  VisitDeclOnlyOnce::Lang_CXX2a);
  EXPECT_EQ(1, Visitor.AutoTypeVisited);
  EXPECT_EQ(1, Visitor.AutoTypeLocVisited);
  EXPECT_EQ(1, Visitor.ConceptDeclsVisited);
  EXPECT_EQ(1, Visitor.ConceptReferencesVisited);
}

} // end anonymous namespace
