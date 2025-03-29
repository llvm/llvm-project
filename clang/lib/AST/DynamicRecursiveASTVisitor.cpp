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
template <bool Const> struct Impl : RecursiveASTVisitor<Impl<Const>> {
  DynamicRecursiveASTVisitorBase<Const> &Visitor;
  Impl(DynamicRecursiveASTVisitorBase<Const> &Visitor) : Visitor(Visitor) {}

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
  bool dataTraverseNode(
      Stmt *S,
      typename RecursiveASTVisitor<Impl>::DataRecursionQueue * = nullptr) {
    // But since we don't support postorder traversal, we don't need it, so
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

template <bool Const> void DynamicRecursiveASTVisitorBase<Const>::anchor() {}

// Helper macros to forward a call to the base implementation since that
// ends up getting very verbose otherwise.

// This calls the RecursiveASTVisitor implementation of the same function,
// stripping any 'const' that the DRAV implementation may have added since
// the RAV implementation largely doesn't use 'const'.
#define FORWARD_TO_BASE(Function, Type, RefOrPointer)                          \
  template <bool Const>                                                        \
  bool DynamicRecursiveASTVisitorBase<Const>::Function(                        \
      MaybeConst<Type> RefOrPointer Param) {                                   \
    return Impl<Const>(*this).RecursiveASTVisitor<Impl<Const>>::Function(      \
        const_cast<Type RefOrPointer>(Param));                                 \
  }

// Same as 'FORWARD_TO_BASE', but doesn't change the parameter type in any way.
#define FORWARD_TO_BASE_EXACT(Function, Type)                                  \
  template <bool Const>                                                        \
  bool DynamicRecursiveASTVisitorBase<Const>::Function(Type Param) {           \
    return Impl<Const>(*this).RecursiveASTVisitor<Impl<Const>>::Function(      \
        Param);                                                                \
  }

FORWARD_TO_BASE(TraverseAST, ASTContext, &)
FORWARD_TO_BASE(TraverseAttr, Attr, *)
FORWARD_TO_BASE(TraverseConstructorInitializer, CXXCtorInitializer, *)
FORWARD_TO_BASE(TraverseDecl, Decl, *)
FORWARD_TO_BASE(TraverseStmt, Stmt, *)
FORWARD_TO_BASE(TraverseNestedNameSpecifier, NestedNameSpecifier, *)
FORWARD_TO_BASE(TraverseTemplateInstantiations, ClassTemplateDecl, *)
FORWARD_TO_BASE(TraverseTemplateInstantiations, VarTemplateDecl, *)
FORWARD_TO_BASE(TraverseTemplateInstantiations, FunctionTemplateDecl, *)
FORWARD_TO_BASE(TraverseConceptRequirement, concepts::Requirement, *)
FORWARD_TO_BASE(TraverseConceptTypeRequirement, concepts::TypeRequirement, *)
FORWARD_TO_BASE(TraverseConceptExprRequirement, concepts::ExprRequirement, *)
FORWARD_TO_BASE(TraverseConceptReference, ConceptReference, *)
FORWARD_TO_BASE(TraverseConceptNestedRequirement,
                concepts::NestedRequirement, *)

FORWARD_TO_BASE_EXACT(TraverseCXXBaseSpecifier, const CXXBaseSpecifier &)
FORWARD_TO_BASE_EXACT(TraverseDeclarationNameInfo, DeclarationNameInfo)
FORWARD_TO_BASE_EXACT(TraverseTemplateArgument, const TemplateArgument &)
FORWARD_TO_BASE_EXACT(TraverseTemplateArguments, ArrayRef<TemplateArgument>)
FORWARD_TO_BASE_EXACT(TraverseTemplateArgumentLoc, const TemplateArgumentLoc &)
FORWARD_TO_BASE_EXACT(TraverseTemplateName, TemplateName)
FORWARD_TO_BASE_EXACT(TraverseType, QualType)
FORWARD_TO_BASE_EXACT(TraverseTypeLoc, TypeLoc)
FORWARD_TO_BASE_EXACT(TraverseTypeConstraint, const TypeConstraint *)
FORWARD_TO_BASE_EXACT(TraverseObjCProtocolLoc, ObjCProtocolLoc)
FORWARD_TO_BASE_EXACT(TraverseNestedNameSpecifierLoc, NestedNameSpecifierLoc)

template <bool Const>
bool DynamicRecursiveASTVisitorBase<Const>::TraverseLambdaCapture(
    MaybeConst<LambdaExpr> *LE, const LambdaCapture *C,
    MaybeConst<Expr> *Init) {
  return Impl<Const>(*this)
      .RecursiveASTVisitor<Impl<Const>>::TraverseLambdaCapture(
          const_cast<LambdaExpr *>(LE), C, const_cast<Expr *>(Init));
}

template <bool Const>
bool DynamicRecursiveASTVisitorBase<Const>::dataTraverseNode(
    MaybeConst<Stmt> *S) {
  return Impl<Const>(*this).RecursiveASTVisitor<Impl<Const>>::dataTraverseNode(
      const_cast<Stmt *>(S), nullptr);
}

// Declare Traverse*() for and friends all concrete Decl classes.
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE)                                                      \
  FORWARD_TO_BASE(Traverse##CLASS##Decl, CLASS##Decl, *)                       \
  FORWARD_TO_BASE(WalkUpFrom##CLASS##Decl, CLASS##Decl, *)
#include "clang/AST/DeclNodes.inc"

// Declare Traverse*() and friends for all concrete Stmt classes.
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT) FORWARD_TO_BASE(Traverse##CLASS, CLASS, *)
#include "clang/AST/StmtNodes.inc"

#define STMT(CLASS, PARENT) FORWARD_TO_BASE(WalkUpFrom##CLASS, CLASS, *)
#include "clang/AST/StmtNodes.inc"

// Declare Traverse*() and friends for all concrete Type classes.
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE)                                                      \
  FORWARD_TO_BASE(Traverse##CLASS##Type, CLASS##Type, *)                       \
  FORWARD_TO_BASE(WalkUpFrom##CLASS##Type, CLASS##Type, *)
#include "clang/AST/TypeNodes.inc"

#define ABSTRACT_TYPELOC(CLASS, BASE)
#define TYPELOC(CLASS, BASE)                                                   \
  FORWARD_TO_BASE_EXACT(Traverse##CLASS##TypeLoc, CLASS##TypeLoc)
#include "clang/AST/TypeLocNodes.def"

#define TYPELOC(CLASS, BASE)                                                   \
  FORWARD_TO_BASE_EXACT(WalkUpFrom##CLASS##TypeLoc, CLASS##TypeLoc)
#include "clang/AST/TypeLocNodes.def"

namespace clang {
template class DynamicRecursiveASTVisitorBase<false>;
template class DynamicRecursiveASTVisitorBase<true>;
} // namespace clang
