//=== DynamicRecursiveASTVisitor.cpp - Dynamic AST Visitor Implementation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an AST visitor that does not require any template
// instantiation to allow users to override its behaviour.
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

  bool shouldTraversePostOrder() const { return false; }

  bool TraverseAST(ASTContext &AST) { return Visitor.TraverseAST(AST); }
  bool TraverseAttr(Attr *At) { return Visitor.TraverseAttr(At); }
  bool TraverseDecl(Decl *D) { return Visitor.TraverseDecl(D); }
  bool TraverseType(QualType T) { return Visitor.TraverseType(T); }
  bool TraverseTypeLoc(TypeLoc TL) { return Visitor.TraverseTypeLoc(TL); }
  bool TraverseStmt(Stmt *S) {
    return Visitor.TraverseStmt(S);
  }

  bool TraverseConstructorInitializer(CXXCtorInitializer *Init) {
    return Visitor.TraverseConstructorInitializer(Init);
  }

  bool TraverseTemplateArgument(const TemplateArgument &Arg) {
    return Visitor.TraverseTemplateArgument(Arg);
  }

  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &ArgLoc) {
    return Visitor.TraverseTemplateArgumentLoc(ArgLoc);
  }

  bool TraverseTemplateName(TemplateName Template) {
    return Visitor.TraverseTemplateName(Template);
  }

  bool TraverseObjCProtocolLoc(ObjCProtocolLoc ProtocolLoc) {
    return Visitor.TraverseObjCProtocolLoc(ProtocolLoc);
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

  bool TraverseConceptReference(ConceptReference *CR) {
    return Visitor.TraverseConceptReference(CR);
  }

  bool TraverseCXXBaseSpecifier(const CXXBaseSpecifier &Base) {
    return Visitor.TraverseCXXBaseSpecifier(Base);
  }

  bool TraverseDeclarationNameInfo(DeclarationNameInfo NameInfo) {
    return Visitor.TraverseDeclarationNameInfo(NameInfo);
  }

  bool TraverseLambdaCapture(LambdaExpr *LE, const LambdaCapture *C,
                             Expr *Init) {
    return Visitor.TraverseLambdaCapture(LE, C, Init);
  }

  bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS) {
    return Visitor.TraverseNestedNameSpecifier(NNS);
  }

  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS) {
    return Visitor.TraverseNestedNameSpecifierLoc(NNS);
  }

  bool VisitConceptReference(ConceptReference *CR) {
    return Visitor.VisitConceptReference(CR);
  }

  bool dataTraverseStmtPre(Stmt *S) { return Visitor.dataTraverseStmtPre(S); }
  bool dataTraverseStmtPost(Stmt *S) { return Visitor.dataTraverseStmtPost(S); }

  // TraverseStmt() always passes in a queue, so we have no choice but to
  // accept it as a parameter here.
  bool dataTraverseNode(Stmt *S, DataRecursionQueue* = nullptr) {
    // But since don't support postorder traversal, we don't need it, so
    // simply discard it here. This way, derived classes don't need to worry
    // about including it as a parameter that they never use.
    return Visitor.dataTraverseNode(S);
  }

  /// Visit a node.
  bool VisitAttr(Attr *A) { return Visitor.VisitAttr(A); }
  bool VisitDecl(Decl *D) { return Visitor.VisitDecl(D); }
  bool VisitStmt(Stmt *S) { return Visitor.VisitStmt(S); }
  bool VisitType(Type *T) { return Visitor.VisitType(T); }
  bool VisitTypeLoc(TypeLoc TL) { return Visitor.VisitTypeLoc(TL); }

  /*#define ATTR_VISITOR_DECLS
  #include "clang/AST/AttrVisitor.inc"
  #undef ATTR_VISITOR_DECLS*/

#define DEF_TRAVERSE_TMPL_INST(kind)                                           \
  bool TraverseTemplateInstantiations(kind##TemplateDecl *D) {                 \
    return Visitor.TraverseTemplateInstantiations(D);                          \
  }
  DEF_TRAVERSE_TMPL_INST(Class)
  DEF_TRAVERSE_TMPL_INST(Var)
  DEF_TRAVERSE_TMPL_INST(Function)
#undef DEF_TRAVERSE_TMPL_INST

  // Declarations.
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE)                                                      \
  bool Traverse##CLASS##Decl(CLASS##Decl *D) {                                 \
    return Visitor.Traverse##CLASS##Decl(D);                                   \
  }
#include "clang/AST/DeclNodes.inc"

#define DECL(CLASS, BASE)                                                      \
  bool Visit##CLASS##Decl(CLASS##Decl *D) {                                    \
    return Visitor.Visit##CLASS##Decl(D);                                      \
  }
#include "clang/AST/DeclNodes.inc"

  // Statements.
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT)                                                    \
  bool Traverse##CLASS(CLASS *S) { return Visitor.Traverse##CLASS(S); }
#include "clang/AST/StmtNodes.inc"

#define STMT(CLASS, PARENT)                                                    \
  bool Visit##CLASS(CLASS *S) { return Visitor.Visit##CLASS(S); }
#include "clang/AST/StmtNodes.inc"

  // Types.
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE)                                                      \
  bool Traverse##CLASS##Type(CLASS##Type *T) {                                 \
    return Visitor.Traverse##CLASS##Type(T);                                   \
  }
#include "clang/AST/TypeNodes.inc"

#define TYPE(CLASS, BASE)                                                      \
  bool Visit##CLASS##Type(CLASS##Type *T) {                                    \
    return Visitor.Visit##CLASS##Type(T);                                      \
  }
#include "clang/AST/TypeNodes.inc"

  // TypeLocs.
#define ABSTRACT_TYPELOC(CLASS, BASE)
#define TYPELOC(CLASS, BASE)                                                   \
  bool Traverse##CLASS##TypeLoc(CLASS##TypeLoc TL) {                           \
    return Visitor.Traverse##CLASS##TypeLoc(TL);                               \
  }
#include "clang/AST/TypeLocNodes.def"

#define TYPELOC(CLASS, BASE)                                                   \
  bool Visit##CLASS##TypeLoc(CLASS##TypeLoc TL) {                              \
    return Visitor.Visit##CLASS##TypeLoc(TL);                                  \
  }
#include "clang/AST/TypeLocNodes.def"
};
} // namespace

void DynamicRecursiveASTVisitor::anchor() { }

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

bool DynamicRecursiveASTVisitor::TraverseLambdaCapture(LambdaExpr *LE,
                                                       const LambdaCapture *C,
                                                       Expr *Init) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseLambdaCapture(LE, C,
                                                                      Init);
}

bool DynamicRecursiveASTVisitor::TraverseStmt(Stmt *S) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseStmt(S);
}

bool DynamicRecursiveASTVisitor::TraverseTemplateArgument(
    const TemplateArgument &Arg) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseTemplateArgument(Arg);
}

bool DynamicRecursiveASTVisitor::TraverseTemplateArguments(
    ArrayRef<TemplateArgument> Args) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseTemplateArguments(Args);
}

bool DynamicRecursiveASTVisitor::TraverseTemplateArgumentLoc(
    const TemplateArgumentLoc &ArgLoc) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseTemplateArgumentLoc(
      ArgLoc);
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
bool DynamicRecursiveASTVisitor::TraverseObjCProtocolLoc(ObjCProtocolLoc ProtocolLoc) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseObjCProtocolLoc(ProtocolLoc);
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

bool DynamicRecursiveASTVisitor::TraverseConceptReference(
    ConceptReference *CR) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseConceptReference(CR);
}


bool DynamicRecursiveASTVisitor::TraverseCXXBaseSpecifier(
    const CXXBaseSpecifier &Base) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseCXXBaseSpecifier(Base);
}

bool DynamicRecursiveASTVisitor::TraverseDeclarationNameInfo(
    DeclarationNameInfo NameInfo) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseDeclarationNameInfo(
      NameInfo);
}

bool DynamicRecursiveASTVisitor::TraverseNestedNameSpecifier(
    NestedNameSpecifier *NNS) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseNestedNameSpecifier(
      NNS);
}

bool DynamicRecursiveASTVisitor::TraverseNestedNameSpecifierLoc(
    NestedNameSpecifierLoc NNS) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseNestedNameSpecifierLoc(
      NNS);
}

bool DynamicRecursiveASTVisitor::dataTraverseNode(Stmt *S) {
  return Impl(*this).RecursiveASTVisitor<Impl>::dataTraverseNode(S, nullptr);
}

/*
#define DYNAMIC_ATTR_VISITOR_IMPL
#include "clang/AST/AttrVisitor.inc"
#undef DYNAMIC_ATTR_VISITOR_IMPL
*/

#define DEF_TRAVERSE_TMPL_INST(kind)                                           \
  bool DynamicRecursiveASTVisitor::TraverseTemplateInstantiations(             \
      kind##TemplateDecl *D) {                                                 \
    return Impl(*this)                                                         \
        .RecursiveASTVisitor<Impl>::TraverseTemplateInstantiations(D);         \
  }
DEF_TRAVERSE_TMPL_INST(Class)
DEF_TRAVERSE_TMPL_INST(Var)
DEF_TRAVERSE_TMPL_INST(Function)
#undef DEF_TRAVERSE_TMPL_INST

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
  }
#include "clang/AST/StmtNodes.inc"

#define STMT(CLASS, PARENT)                                                    \
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
  }
#include "clang/AST/TypeLocNodes.def"

#define TYPELOC(CLASS, BASE)                                                   \
  bool DynamicRecursiveASTVisitor::WalkUpFrom##CLASS##TypeLoc(                 \
      CLASS##TypeLoc TL) {                                                     \
    return Impl(*this).RecursiveASTVisitor<Impl>::WalkUpFrom##CLASS##TypeLoc(  \
        TL);                                                                   \
  }
#include "clang/AST/TypeLocNodes.def"
