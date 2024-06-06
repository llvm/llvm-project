//=== DynamicRecursiveASTVisitor.cpp - Dynamic AST Visitor Implementation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an AST visitor that does not require any template
// instantiation to allow users to override this behaviour.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/RecursiveASTVisitor.h"

using namespace clang;

namespace {
struct Impl : RecursiveASTVisitor<Impl> {
  DynamicRecursiveASTVisitor &Visitor;
  Impl(DynamicRecursiveASTVisitor &Visitor) : Visitor(Visitor) {}

  bool shouldVisitTemplateInstantiations() const {
    return Visitor.ShouldVisitTemplateInstantiations;
  }

  bool shouldWalkTypesOfTypeLocs() const {
    return Visitor.ShouldWalkTypesOfTypeLocs;
  }

  bool shouldVisitImplicitCode() const {
    return Visitor.ShouldVisitImplicitCode;
  }

  bool shouldVisitLambdaBody() const { return Visitor.ShouldVisitLambdaBody; }

  bool shouldTraversePostOrder() const {
    return Visitor.ShouldTraversePostOrder;
  }

  bool TraverseAST(ASTContext &AST) { return Visitor.TraverseAST(AST); }
  bool TraverseAttr(Attr *At) { return Visitor.TraverseAttr(At); }
  bool TraverseDecl(Decl *D) { return Visitor.TraverseDecl(D); }
  bool TraverseType(QualType T) { return Visitor.TraverseType(T); }
  bool TraverseTypeLoc(TypeLoc TL) { return Visitor.TraverseTypeLoc(TL); }
  bool TraverseStmt(Stmt *S, DataRecursionQueue *Queue = nullptr) {
    return Visitor.TraverseStmt(S, Queue);
  }

  bool TraverseConstructorInitializer(CXXCtorInitializer *Init) {
    return Visitor.TraverseConstructorInitializer(Init);
  }

  bool TraverseTemplateArgument(const TemplateArgument &Arg) {
    return Visitor.TraverseTemplateArgument(Arg);
  }

  bool TraverseTemplateName(TemplateName Template) {
    return Visitor.TraverseTemplateName(Template);
  }

  bool TraverseTypeConstraint(const TypeConstraint *C) {
    return Visitor.TraverseTypeConstraint(C);
  }
  bool TraverseConceptRequirement(concepts::Requirement *R) {
    return Visitor.TraverseConceptRequirement(R);
  }
  bool TraverseConceptTypeRequirement(concepts::TypeRequirement *R) {
    return Visitor.TraverseConceptTypeRequirement(R);
  }
  bool TraverseConceptExprRequirement(concepts::ExprRequirement *R) {
    return Visitor.TraverseConceptExprRequirement(R);
  }
  bool TraverseConceptNestedRequirement(concepts::NestedRequirement *R) {
    return Visitor.TraverseConceptNestedRequirement(R);
  }

  /// Visit a node.
  bool VisitAttr(Attr *A) { return Visitor.VisitAttr(A); }
  bool VisitDecl(Decl *D) { return Visitor.VisitDecl(D); }
  bool VisitStmt(Stmt *S) { return Visitor.VisitStmt(S); }
  bool VisitType(Type *T) { return Visitor.VisitType(T); }
  bool VisitTypeLoc(TypeLoc TL) { return Visitor.VisitTypeLoc(TL); }

  /// Walk up from a node.
  bool WalkUpFromDecl(Decl *D) { return Visitor.WalkUpFromDecl(D); }
  bool WalkUpFromStmt(Stmt *S) { return Visitor.WalkUpFromStmt(S); }
  bool WalkUpFromType(Type *T) { return Visitor.WalkUpFromType(T); }
  bool WalkUpFromTypeLoc(TypeLoc TL) { return Visitor.WalkUpFromTypeLoc(TL); }

  /*#define ATTR_VISITOR_DECLS
  #include "clang/AST/AttrVisitor.inc"
  #undef ATTR_VISITOR_DECLS*/

#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE)                                                      \
  bool Traverse##CLASS##Decl(CLASS##Decl *D) {                                 \
    return Visitor.Traverse##CLASS##Decl(D);                                   \
  }                                                                            \
  bool WalkUpFrom##CLASS##Decl(CLASS##Decl *D) {                               \
    return Visitor.WalkUpFrom##CLASS##Decl(D);                                 \
  }                                                                            \
  bool Visit##CLASS##Decl(CLASS##Decl *D) {                                    \
    return Visitor.Visit##CLASS##Decl(D);                                      \
  }
#include "clang/AST/DeclNodes.inc"

#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT)                                                    \
  bool Traverse##CLASS(CLASS *S) { return Visitor.Traverse##CLASS(S); }        \
  bool WalkUpFrom##CLASS(CLASS *S) { return Visitor.WalkUpFrom##CLASS(S); }    \
  bool Visit##CLASS(CLASS *S) { return Visitor.Visit##CLASS(S); }
#include "clang/AST/StmtNodes.inc"

  // Declare Traverse*() and friends for all concrete Typeclasses.
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE)                                                      \
  bool Traverse##CLASS##Type(CLASS##Type *T) {                                 \
    return Visitor.Traverse##CLASS##Type(T);                                   \
  }                                                                            \
  bool WalkUpFrom##CLASS##Type(CLASS##Type *T) {                               \
    return Visitor.WalkUpFrom##CLASS##Type(T);                                 \
  }                                                                            \
  bool Visit##CLASS##Type(CLASS##Type *T) {                                    \
    return Visitor.Visit##CLASS##Type(T);                                      \
  }
#include "clang/AST/TypeNodes.inc"

#define ABSTRACT_TYPELOC(CLASS, BASE)
#define TYPELOC(CLASS, BASE)                                                   \
  bool Traverse##CLASS##TypeLoc(CLASS##TypeLoc TL) {                           \
    return Visitor.Traverse##CLASS##TypeLoc(TL);                               \
  }                                                                            \
  bool WalkUpFrom##CLASS##TypeLoc(CLASS##TypeLoc TL) {                         \
    return Visitor.WalkUpFrom##CLASS##TypeLoc(TL);                             \
  }                                                                            \
  bool Visit##CLASS##TypeLoc(CLASS##TypeLoc TL) {                              \
    return Visitor.Visit##CLASS##TypeLoc(TL);                                  \
  }
#include "clang/AST/TypeLocNodes.def"
};
} // namespace

// Declared out of line to serve as a vtable anchor.
DynamicRecursiveASTVisitor::~DynamicRecursiveASTVisitor() = default;

bool DynamicRecursiveASTVisitor::TraverseAST(ASTContext &AST) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseAST(AST);
}

bool DynamicRecursiveASTVisitor::TraverseAttr(Attr *At) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseAttr(At);
}

bool DynamicRecursiveASTVisitor::TraverseConstructorInitializer(
    CXXCtorInitializer *Init) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseConstructorInitializer(
      Init);
}

bool DynamicRecursiveASTVisitor::TraverseDecl(Decl *D) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseDecl(D);
}

bool DynamicRecursiveASTVisitor::TraverseStmt(Stmt *S,
                                              DataRecursionQueue *Queue) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseStmt(S, Queue);
}

bool DynamicRecursiveASTVisitor::TraverseTemplateArgument(
    const TemplateArgument &Arg) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseTemplateArgument(Arg);
}

bool DynamicRecursiveASTVisitor::TraverseTemplateArguments(
    ArrayRef<TemplateArgument> Args) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseTemplateArguments(Args);
}

bool DynamicRecursiveASTVisitor::TraverseTemplateName(TemplateName Template) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseTemplateName(Template);
}

bool DynamicRecursiveASTVisitor::TraverseType(QualType T) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseType(T);
}

bool DynamicRecursiveASTVisitor::TraverseTypeLoc(TypeLoc TL) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseTypeLoc(TL);
}

bool DynamicRecursiveASTVisitor::TraverseTypeConstraint(
    const TypeConstraint *C) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseTypeConstraint(C);
}
bool DynamicRecursiveASTVisitor::TraverseConceptRequirement(
    concepts::Requirement *R) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseConceptRequirement(R);
}
bool DynamicRecursiveASTVisitor::TraverseConceptTypeRequirement(
    concepts::TypeRequirement *R) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseConceptTypeRequirement(
      R);
}
bool DynamicRecursiveASTVisitor::TraverseConceptExprRequirement(
    concepts::ExprRequirement *R) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseConceptExprRequirement(
      R);
}
bool DynamicRecursiveASTVisitor::TraverseConceptNestedRequirement(
    concepts::NestedRequirement *R) {
  return Impl(*this)
      .RecursiveASTVisitor<Impl>::TraverseConceptNestedRequirement(R);
}

/*
#define DYNAMIC_ATTR_VISITOR_IMPL
#include "clang/AST/AttrVisitor.inc"
#undef DYNAMIC_ATTR_VISITOR_IMPL
*/

// Declare Traverse*() for and friends all concrete Decl classes.
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE)                                                      \
  bool DynamicRecursiveASTVisitor::Traverse##CLASS##Decl(CLASS##Decl *D) {     \
    return Impl(*this).RecursiveASTVisitor<Impl>::Traverse##CLASS##Decl(D);    \
  }                                                                            \
  bool DynamicRecursiveASTVisitor::WalkUpFrom##CLASS##Decl(CLASS##Decl *D) {   \
    return Impl(*this).RecursiveASTVisitor<Impl>::WalkUpFrom##CLASS##Decl(D);  \
  }
#include "clang/AST/DeclNodes.inc"

// Declare Traverse*() and friends for all concrete Stmt classes.
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT)                                                    \
  bool DynamicRecursiveASTVisitor::Traverse##CLASS(CLASS *S) {                 \
    return Impl(*this).RecursiveASTVisitor<Impl>::Traverse##CLASS(S);          \
  }                                                                            \
  bool DynamicRecursiveASTVisitor::WalkUpFrom##CLASS(CLASS *S) {               \
    return Impl(*this).RecursiveASTVisitor<Impl>::WalkUpFrom##CLASS(S);        \
  }
#include "clang/AST/StmtNodes.inc"

// Declare Traverse*() and friends for all concrete Typeclasses.
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE)                                                      \
  bool DynamicRecursiveASTVisitor::Traverse##CLASS##Type(CLASS##Type *T) {     \
    return Impl(*this).RecursiveASTVisitor<Impl>::Traverse##CLASS##Type(T);    \
  }                                                                            \
  bool DynamicRecursiveASTVisitor::WalkUpFrom##CLASS##Type(CLASS##Type *T) {   \
    return Impl(*this).RecursiveASTVisitor<Impl>::WalkUpFrom##CLASS##Type(T);  \
  }
#include "clang/AST/TypeNodes.inc"

#define ABSTRACT_TYPELOC(CLASS, BASE)
#define TYPELOC(CLASS, BASE)                                                   \
  bool DynamicRecursiveASTVisitor::Traverse##CLASS##TypeLoc(                   \
      CLASS##TypeLoc TL) {                                                     \
    return Impl(*this).RecursiveASTVisitor<Impl>::Traverse##CLASS##TypeLoc(    \
        TL);                                                                   \
  }                                                                            \
  bool DynamicRecursiveASTVisitor::WalkUpFrom##CLASS##TypeLoc(                 \
      CLASS##TypeLoc TL) {                                                     \
    return Impl(*this).RecursiveASTVisitor<Impl>::WalkUpFrom##CLASS##TypeLoc(  \
        TL);                                                                   \
  }
#include "clang/AST/TypeLocNodes.def"
