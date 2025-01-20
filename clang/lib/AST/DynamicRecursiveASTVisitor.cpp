//=== DynamicRecursiveASTVisitor.cpp - Dynamic AST Visitor Implementation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements DynamicRecursiveASTVisitor in terms of the CRTP-based
// RecursiveASTVisitor.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/RecursiveASTVisitor.h"

using namespace clang;

// The implementation of DRAV deserves some explanation:
//
// We want to implement DynamicRecursiveASTVisitor without having to inherit or
// reference RecursiveASTVisitor in any way in the header: if we instantiate
// RAV in the header, then every user of (or rather every file that uses) DRAV
// still has to instantiate a RAV, which gets us nowhere. Moreover, even just
// including RecursiveASTVisitor.h would probably cause some amount of slowdown
// because we'd have to parse a huge template. For these reasons, the fact that
// DRAV is implemented using a RAV is solely an implementation detail.
//
// As for the implementation itself, DRAV by default acts exactly like a RAV
// that overrides none of RAV's functions. There are two parts to this:
//
//   1. Any function in DRAV has to act like the corresponding function in RAV,
//      unless overridden by a derived class, of course.
//
//   2. Any call to a function by the RAV implementation that DRAV allows to be
//      overridden must be transformed to a virtual call on the user-provided
//      DRAV object: if some function in RAV calls e.g. TraverseCallExpr()
//      during traversal, then the derived class's TraverseCallExpr() must be
//      called (provided it overrides TraverseCallExpr()).
//
// The 'Impl' class is a helper that connects the two implementations; it is
// a wrapper around a reference to a DRAV that is itself a RecursiveASTVisitor.
// It overrides every function in RAV *that is virtual in DRAV* to perform a
// virtual call on its DRAV reference. This accomplishes point 2 above.
//
// Point 1 is accomplished by, first, having the base class implementation of
// each of the virtual functions construct an Impl object (which is actually
// just a no-op), passing in itself so that any virtual calls use the right
// vtable. Secondly, it then calls RAV's implementation of that same function
// *on Impl* (using a qualified call so that we actually call into the RAV
// implementation instead of Impl's version of that same function); this way,
// we both execute RAV's implementation for this function only and ensure that
// calls to subsequent functions call into Impl via CRTP (and Impl then calls
// back into DRAV and so on).
//
// While this ends up constructing a lot of Impl instances (almost one per
// function call), this doesn't really matter since Impl just holds a single
// pointer, and everything in this file should get inlined into all the DRAV
// functions here anyway.
//
//===----------------------------------------------------------------------===//
//
// The following illustrates how a call to an (overridden) function is actually
// resolved: given some class 'Derived' that derives from DRAV and overrides
// TraverseStmt(), if we are traversing some AST, and TraverseStmt() is called
// by the RAV implementation, the following happens:
//
//   1. Impl::TraverseStmt() overrides RAV::TraverseStmt() via CRTP, so the
//      former is called.
//
//   2. Impl::TraverseStmt() performs a virtual call to the visitor (which is
//      an instance to Derived), so Derived::TraverseStmt() is called.
//
//   End result: Derived::TraverseStmt() is executed.
//
// Suppose some other function, e.g. TraverseCallExpr(), which is NOT overridden
// by Derived is called, we get:
//
//   1. Impl::TraverseCallExpr() overrides RAV::TraverseCallExpr() via CRTP,
//      so the former is called.
//
//   2. Impl::TraverseCallExpr() performs a virtual call, but since Derived
//      does not override that function, DRAV::TraverseCallExpr() is called.
//
//   3. DRAV::TraverseCallExpr() creates a new instance of Impl, passing in
//      itself (this doesn't change that the pointer is an instance of Derived);
//      it then calls RAV::TraverseCallExpr() on the Impl object, which actually
//      ends up executing RAV's implementation because we used a qualified
//      function call.
//
//   End result: RAV::TraverseCallExpr() is executed,
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

  // Supporting post-order would be very hard because of quirks of the
  // RAV implementation that only work with CRTP. It also is only used
  // by less than 5 visitors in the entire code base.
  bool shouldTraversePostOrder() const { return false; }

  bool TraverseAST(ASTContext &AST) { return Visitor.TraverseAST(AST); }
  bool TraverseAttr(Attr *At) { return Visitor.TraverseAttr(At); }
  bool TraverseDecl(Decl *D) { return Visitor.TraverseDecl(D); }
  bool TraverseType(QualType T) { return Visitor.TraverseType(T); }
  bool TraverseTypeLoc(TypeLoc TL) { return Visitor.TraverseTypeLoc(TL); }
  bool TraverseStmt(Stmt *S) { return Visitor.TraverseStmt(S); }

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
  bool dataTraverseNode(Stmt *S, DataRecursionQueue * = nullptr) {
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

#define DEF_TRAVERSE_TMPL_INST(kind)                                           \
  bool TraverseTemplateInstantiations(kind##TemplateDecl *D) {                 \
    return Visitor.TraverseTemplateInstantiations(D);                          \
  }
  DEF_TRAVERSE_TMPL_INST(Class)
  DEF_TRAVERSE_TMPL_INST(Var)
  DEF_TRAVERSE_TMPL_INST(Function)
#undef DEF_TRAVERSE_TMPL_INST

  // Decls.
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

  // Stmts.
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

void DynamicRecursiveASTVisitor::anchor() {}

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
bool DynamicRecursiveASTVisitor::TraverseObjCProtocolLoc(
    ObjCProtocolLoc ProtocolLoc) {
  return Impl(*this).RecursiveASTVisitor<Impl>::TraverseObjCProtocolLoc(
      ProtocolLoc);
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
