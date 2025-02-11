//===--- DynamicRecursiveASTVisitor.h - Virtual AST Visitor -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DynamicRecursiveASTVisitor interface, which acts
//  identically to RecursiveASTVisitor, except that it uses virtual dispatch
//  instead of CRTP, which greatly improves compile times and binary size.
//
//  Prefer to use this over RecursiveASTVisitor whenever possible.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_DYNAMIC_RECURSIVE_AST_VISITOR_H
#define LLVM_CLANG_AST_DYNAMIC_RECURSIVE_AST_VISITOR_H

#include "clang/AST/Attr.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/TypeLoc.h"

namespace clang {
class ASTContext;

/// Recursive AST visitor that supports extension via dynamic dispatch.
///
/// Like RecursiveASTVisitor, this class allows for traversal of arbitrarily
/// complex ASTs. The main difference is that this uses virtual functions
/// instead of CRTP, which greatly improves compile times of Clang itself,
/// as well as binary size.
///
/// Instead of functions (e.g. shouldVisitImplicitCode()), this class
/// uses member variables (e.g. ShouldVisitImplicitCode) to control
/// visitation behaviour.
///
/// However, there is no support for overriding some of the less commonly
/// used features of the RAV, such as WalkUpFromX or attribute traversal
/// (attributes can still be traversed, but you can't change what happens
/// when we traverse one).
///
/// The following is a list of RAV features that are NOT customisable:
///
///   - Visiting attributes,
///   - Overriding WalkUpFromX,
///   - Overriding getStmtChildren().
///
/// Furthermore, post-order traversal is not supported at all.
///
/// Prefer to use this over RecursiveASTVisitor unless you absolutely
/// need to use one of the features listed above (e.g. overriding
/// WalkUpFromX or post-order traversal).
///
/// \see RecursiveASTVisitor.
class DynamicRecursiveASTVisitor {
public:
  /// Whether this visitor should recurse into template instantiations.
  bool ShouldVisitTemplateInstantiations = false;

  /// Whether this visitor should recurse into the types of TypeLocs.
  bool ShouldWalkTypesOfTypeLocs = true;

  /// Whether this visitor should recurse into implicit code, e.g.
  /// implicit constructors and destructors.
  bool ShouldVisitImplicitCode = false;

  /// Whether this visitor should recurse into lambda body.
  bool ShouldVisitLambdaBody = true;

protected:
  DynamicRecursiveASTVisitor() = default;
  DynamicRecursiveASTVisitor(DynamicRecursiveASTVisitor &&) = default;
  DynamicRecursiveASTVisitor(const DynamicRecursiveASTVisitor &) = default;
  DynamicRecursiveASTVisitor &
  operator=(DynamicRecursiveASTVisitor &&) = default;
  DynamicRecursiveASTVisitor &
  operator=(const DynamicRecursiveASTVisitor &) = default;

public:
  virtual void anchor();
  virtual ~DynamicRecursiveASTVisitor() = default;

  /// Recursively visits an entire AST, starting from the TranslationUnitDecl.
  /// \returns false if visitation was terminated early.
  virtual bool TraverseAST(ASTContext &AST);

  /// Recursively visit an attribute, by dispatching to
  /// Traverse*Attr() based on the argument's dynamic type.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is a Null type location).
  virtual bool TraverseAttr(Attr *At);

  /// Recursively visit a constructor initializer.  This
  /// automatically dispatches to another visitor for the initializer
  /// expression, but not for the name of the initializer, so may
  /// be overridden for clients that need access to the name.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseConstructorInitializer(CXXCtorInitializer *Init);

  /// Recursively visit a base specifier. This can be overridden by a
  /// subclass.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseCXXBaseSpecifier(const CXXBaseSpecifier &Base);

  /// Recursively visit a declaration, by dispatching to
  /// Traverse*Decl() based on the argument's dynamic type.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is NULL).
  virtual bool TraverseDecl(Decl *D);

  /// Recursively visit a name with its location information.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseDeclarationNameInfo(DeclarationNameInfo NameInfo);

  /// Recursively visit a lambda capture. \c Init is the expression that
  /// will be used to initialize the capture.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseLambdaCapture(LambdaExpr *LE, const LambdaCapture *C,
                                     Expr *Init);

  /// Recursively visit a C++ nested-name-specifier.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS);

  /// Recursively visit a C++ nested-name-specifier with location
  /// information.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS);

  /// Recursively visit a statement or expression, by
  /// dispatching to Traverse*() based on the argument's dynamic type.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is nullptr).
  virtual bool TraverseStmt(Stmt *S);

  /// Recursively visit a template argument and dispatch to the
  /// appropriate method for the argument type.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  // FIXME: migrate callers to TemplateArgumentLoc instead.
  virtual bool TraverseTemplateArgument(const TemplateArgument &Arg);

  /// Recursively visit a template argument location and dispatch to the
  /// appropriate method for the argument type.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &ArgLoc);

  /// Recursively visit a set of template arguments.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  // FIXME: take a TemplateArgumentLoc* (or TemplateArgumentListInfo) instead.
  // Not virtual for now because no-one overrides it.
  bool TraverseTemplateArguments(ArrayRef<TemplateArgument> Args);

  /// Recursively visit a template name and dispatch to the
  /// appropriate method.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseTemplateName(TemplateName Template);

  /// Recursively visit a type, by dispatching to
  /// Traverse*Type() based on the argument's getTypeClass() property.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is a Null type).
  virtual bool TraverseType(QualType T);

  /// Recursively visit a type with location, by dispatching to
  /// Traverse*TypeLoc() based on the argument type's getTypeClass() property.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is a Null type location).
  virtual bool TraverseTypeLoc(TypeLoc TL);

  /// Recursively visit an Objective-C protocol reference with location
  /// information.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseObjCProtocolLoc(ObjCProtocolLoc ProtocolLoc);

  /// Traverse a concept (requirement).
  virtual bool TraverseTypeConstraint(const TypeConstraint *C);
  virtual bool TraverseConceptRequirement(concepts::Requirement *R);
  virtual bool TraverseConceptTypeRequirement(concepts::TypeRequirement *R);
  virtual bool TraverseConceptExprRequirement(concepts::ExprRequirement *R);
  virtual bool TraverseConceptNestedRequirement(concepts::NestedRequirement *R);
  virtual bool TraverseConceptReference(ConceptReference *CR);
  virtual bool VisitConceptReference(ConceptReference *CR) { return true; }

  /// Visit a node.
  virtual bool VisitAttr(Attr *A) { return true; }
  virtual bool VisitDecl(Decl *D) { return true; }
  virtual bool VisitStmt(Stmt *S) { return true; }
  virtual bool VisitType(Type *T) { return true; }
  virtual bool VisitTypeLoc(TypeLoc TL) { return true; }

  /// Walk up from a node.
  bool WalkUpFromDecl(Decl *D) { return VisitDecl(D); }
  bool WalkUpFromStmt(Stmt *S) { return VisitStmt(S); }
  bool WalkUpFromType(Type *T) { return VisitType(T); }
  bool WalkUpFromTypeLoc(TypeLoc TL) { return VisitTypeLoc(TL); }

  /// Invoked before visiting a statement or expression via data recursion.
  ///
  /// \returns false to skip visiting the node, true otherwise.
  virtual bool dataTraverseStmtPre(Stmt *S) { return true; }

  /// Invoked after visiting a statement or expression via data recursion.
  /// This is not invoked if the previously invoked \c dataTraverseStmtPre
  /// returned false.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool dataTraverseStmtPost(Stmt *S) { return true; }
  virtual bool dataTraverseNode(Stmt *S);

#define DEF_TRAVERSE_TMPL_INST(kind)                                           \
  virtual bool TraverseTemplateInstantiations(kind##TemplateDecl *D);
  DEF_TRAVERSE_TMPL_INST(Class)
  DEF_TRAVERSE_TMPL_INST(Var)
  DEF_TRAVERSE_TMPL_INST(Function)
#undef DEF_TRAVERSE_TMPL_INST

  // Decls.
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE) virtual bool Traverse##CLASS##Decl(CLASS##Decl *D);
#include "clang/AST/DeclNodes.inc"

#define DECL(CLASS, BASE)                                                      \
  bool WalkUpFrom##CLASS##Decl(CLASS##Decl *D);                                \
  virtual bool Visit##CLASS##Decl(CLASS##Decl *D) { return true; }
#include "clang/AST/DeclNodes.inc"

  // Stmts.
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT) virtual bool Traverse##CLASS(CLASS *S);
#include "clang/AST/StmtNodes.inc"

#define STMT(CLASS, PARENT)                                                    \
  bool WalkUpFrom##CLASS(CLASS *S);                                            \
  virtual bool Visit##CLASS(CLASS *S) { return true; }
#include "clang/AST/StmtNodes.inc"

  // Types.
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE) virtual bool Traverse##CLASS##Type(CLASS##Type *T);
#include "clang/AST/TypeNodes.inc"

#define TYPE(CLASS, BASE)                                                      \
  bool WalkUpFrom##CLASS##Type(CLASS##Type *T);                                \
  virtual bool Visit##CLASS##Type(CLASS##Type *T) { return true; }
#include "clang/AST/TypeNodes.inc"

  // TypeLocs.
#define ABSTRACT_TYPELOC(CLASS, BASE)
#define TYPELOC(CLASS, BASE)                                                   \
  virtual bool Traverse##CLASS##TypeLoc(CLASS##TypeLoc TL);
#include "clang/AST/TypeLocNodes.def"

#define TYPELOC(CLASS, BASE)                                                   \
  bool WalkUpFrom##CLASS##TypeLoc(CLASS##TypeLoc TL);                          \
  virtual bool Visit##CLASS##TypeLoc(CLASS##TypeLoc TL) { return true; }
#include "clang/AST/TypeLocNodes.def"
};
} // namespace clang

#endif // LLVM_CLANG_AST_DYNAMIC_RECURSIVE_AST_VISITOR_H
