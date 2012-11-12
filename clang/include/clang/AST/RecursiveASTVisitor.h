//===--- RecursiveASTVisitor.h - Recursive AST Visitor ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the RecursiveASTVisitor interface, which recursively
//  traverses the entire AST.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_RECURSIVEASTVISITOR_H
#define LLVM_CLANG_AST_RECURSIVEASTVISITOR_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"

// The following three macros are used for meta programming.  The code
// using them is responsible for defining macro OPERATOR().

// All unary operators.
#define UNARYOP_LIST()                          \
  OPERATOR(PostInc)   OPERATOR(PostDec)         \
  OPERATOR(PreInc)    OPERATOR(PreDec)          \
  OPERATOR(AddrOf)    OPERATOR(Deref)           \
  OPERATOR(Plus)      OPERATOR(Minus)           \
  OPERATOR(Not)       OPERATOR(LNot)            \
  OPERATOR(Real)      OPERATOR(Imag)            \
  OPERATOR(Extension)

// All binary operators (excluding compound assign operators).
#define BINOP_LIST() \
  OPERATOR(PtrMemD)              OPERATOR(PtrMemI)    \
  OPERATOR(Mul)   OPERATOR(Div)  OPERATOR(Rem)        \
  OPERATOR(Add)   OPERATOR(Sub)  OPERATOR(Shl)        \
  OPERATOR(Shr)                                       \
                                                      \
  OPERATOR(LT)    OPERATOR(GT)   OPERATOR(LE)         \
  OPERATOR(GE)    OPERATOR(EQ)   OPERATOR(NE)         \
  OPERATOR(And)   OPERATOR(Xor)  OPERATOR(Or)         \
  OPERATOR(LAnd)  OPERATOR(LOr)                       \
                                                      \
  OPERATOR(Assign)                                    \
  OPERATOR(Comma)

// All compound assign operators.
#define CAO_LIST()                                                      \
  OPERATOR(Mul) OPERATOR(Div) OPERATOR(Rem) OPERATOR(Add) OPERATOR(Sub) \
  OPERATOR(Shl) OPERATOR(Shr) OPERATOR(And) OPERATOR(Or)  OPERATOR(Xor)

namespace clang {

// A helper macro to implement short-circuiting when recursing.  It
// invokes CALL_EXPR, which must be a method call, on the derived
// object (s.t. a user of RecursiveASTVisitor can override the method
// in CALL_EXPR).
#define TRY_TO(CALL_EXPR) \
  do { if (!getDerived().CALL_EXPR) return false; } while (0)

/// \brief A class that does preorder depth-first traversal on the
/// entire Clang AST and visits each node.
///
/// This class performs three distinct tasks:
///   1. traverse the AST (i.e. go to each node);
///   2. at a given node, walk up the class hierarchy, starting from
///      the node's dynamic type, until the top-most class (e.g. Stmt,
///      Decl, or Type) is reached.
///   3. given a (node, class) combination, where 'class' is some base
///      class of the dynamic type of 'node', call a user-overridable
///      function to actually visit the node.
///
/// These tasks are done by three groups of methods, respectively:
///   1. TraverseDecl(Decl *x) does task #1.  It is the entry point
///      for traversing an AST rooted at x.  This method simply
///      dispatches (i.e. forwards) to TraverseFoo(Foo *x) where Foo
///      is the dynamic type of *x, which calls WalkUpFromFoo(x) and
///      then recursively visits the child nodes of x.
///      TraverseStmt(Stmt *x) and TraverseType(QualType x) work
///      similarly.
///   2. WalkUpFromFoo(Foo *x) does task #2.  It does not try to visit
///      any child node of x.  Instead, it first calls WalkUpFromBar(x)
///      where Bar is the direct parent class of Foo (unless Foo has
///      no parent), and then calls VisitFoo(x) (see the next list item).
///   3. VisitFoo(Foo *x) does task #3.
///
/// These three method groups are tiered (Traverse* > WalkUpFrom* >
/// Visit*).  A method (e.g. Traverse*) may call methods from the same
/// tier (e.g. other Traverse*) or one tier lower (e.g. WalkUpFrom*).
/// It may not call methods from a higher tier.
///
/// Note that since WalkUpFromFoo() calls WalkUpFromBar() (where Bar
/// is Foo's super class) before calling VisitFoo(), the result is
/// that the Visit*() methods for a given node are called in the
/// top-down order (e.g. for a node of type NamedDecl, the order will
/// be VisitDecl(), VisitNamedDecl(), and then VisitNamespaceDecl()).
///
/// This scheme guarantees that all Visit*() calls for the same AST
/// node are grouped together.  In other words, Visit*() methods for
/// different nodes are never interleaved.
///
/// Clients of this visitor should subclass the visitor (providing
/// themselves as the template argument, using the curiously recurring
/// template pattern) and override any of the Traverse*, WalkUpFrom*,
/// and Visit* methods for declarations, types, statements,
/// expressions, or other AST nodes where the visitor should customize
/// behavior.  Most users only need to override Visit*.  Advanced
/// users may override Traverse* and WalkUpFrom* to implement custom
/// traversal strategies.  Returning false from one of these overridden
/// functions will abort the entire traversal.
///
/// By default, this visitor tries to visit every part of the explicit
/// source code exactly once.  The default policy towards templates
/// is to descend into the 'pattern' class or function body, not any
/// explicit or implicit instantiations.  Explicit specializations
/// are still visited, and the patterns of partial specializations
/// are visited separately.  This behavior can be changed by
/// overriding shouldVisitTemplateInstantiations() in the derived class
/// to return true, in which case all known implicit and explicit
/// instantiations will be visited at the same time as the pattern
/// from which they were produced.
template<typename Derived>
class RecursiveASTVisitor {
public:
  /// \brief Return a reference to the derived class.
  Derived &getDerived() { return *static_cast<Derived*>(this); }

  /// \brief Return whether this visitor should recurse into
  /// template instantiations.
  bool shouldVisitTemplateInstantiations() const { return false; }

  /// \brief Return whether this visitor should recurse into the types of
  /// TypeLocs.
  bool shouldWalkTypesOfTypeLocs() const { return true; }

  /// \brief Return whether this visitor should recurse into implicit
  /// code, e.g., implicit constructors and destructors.
  bool shouldVisitImplicitCode() const { return false; }

  /// \brief Return whether \param S should be traversed using data recursion
  /// to avoid a stack overflow with extreme cases.
  bool shouldUseDataRecursionFor(Stmt *S) const {
    return isa<BinaryOperator>(S) || isa<UnaryOperator>(S) ||
           isa<CaseStmt>(S) || isa<CXXOperatorCallExpr>(S);
  }

  /// \brief Recursively visit a statement or expression, by
  /// dispatching to Traverse*() based on the argument's dynamic type.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is NULL).
  bool TraverseStmt(Stmt *S);

  /// \brief Recursively visit a type, by dispatching to
  /// Traverse*Type() based on the argument's getTypeClass() property.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is a Null type).
  bool TraverseType(QualType T);

  /// \brief Recursively visit a type with location, by dispatching to
  /// Traverse*TypeLoc() based on the argument type's getTypeClass() property.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is a Null type location).
  bool TraverseTypeLoc(TypeLoc TL);

  /// \brief Recursively visit a declaration, by dispatching to
  /// Traverse*Decl() based on the argument's dynamic type.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is NULL).
  bool TraverseDecl(Decl *D);

  /// \brief Recursively visit a C++ nested-name-specifier.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS);

  /// \brief Recursively visit a C++ nested-name-specifier with location
  /// information.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS);

  /// \brief Recursively visit a name with its location information.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  bool TraverseDeclarationNameInfo(DeclarationNameInfo NameInfo);

  /// \brief Recursively visit a template name and dispatch to the
  /// appropriate method.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  bool TraverseTemplateName(TemplateName Template);

  /// \brief Recursively visit a template argument and dispatch to the
  /// appropriate method for the argument type.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  // FIXME: migrate callers to TemplateArgumentLoc instead.
  bool TraverseTemplateArgument(const TemplateArgument &Arg);

  /// \brief Recursively visit a template argument location and dispatch to the
  /// appropriate method for the argument type.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &ArgLoc);

  /// \brief Recursively visit a set of template arguments.
  /// This can be overridden by a subclass, but it's not expected that
  /// will be needed -- this visitor always dispatches to another.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  // FIXME: take a TemplateArgumentLoc* (or TemplateArgumentListInfo) instead.
  bool TraverseTemplateArguments(const TemplateArgument *Args,
                                 unsigned NumArgs);

  /// \brief Recursively visit a constructor initializer.  This
  /// automatically dispatches to another visitor for the initializer
  /// expression, but not for the name of the initializer, so may
  /// be overridden for clients that need access to the name.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  bool TraverseConstructorInitializer(CXXCtorInitializer *Init);

  /// \brief Recursively visit a lambda capture.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  bool TraverseLambdaCapture(LambdaExpr::Capture C);
  
  // ---- Methods on Stmts ----

  // Declare Traverse*() for all concrete Stmt classes.
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT)                                     \
  bool Traverse##CLASS(CLASS *S);
#include "clang/AST/StmtNodes.inc"
  // The above header #undefs ABSTRACT_STMT and STMT upon exit.

  // Define WalkUpFrom*() and empty Visit*() for all Stmt classes.
  bool WalkUpFromStmt(Stmt *S) { return getDerived().VisitStmt(S); }
  bool VisitStmt(Stmt *S) { return true; }
#define STMT(CLASS, PARENT)                                     \
  bool WalkUpFrom##CLASS(CLASS *S) {                            \
    TRY_TO(WalkUpFrom##PARENT(S));                              \
    TRY_TO(Visit##CLASS(S));                                    \
    return true;                                                \
  }                                                             \
  bool Visit##CLASS(CLASS *S) { return true; }
#include "clang/AST/StmtNodes.inc"

  // Define Traverse*(), WalkUpFrom*(), and Visit*() for unary
  // operator methods.  Unary operators are not classes in themselves
  // (they're all opcodes in UnaryOperator) but do have visitors.
#define OPERATOR(NAME)                                           \
  bool TraverseUnary##NAME(UnaryOperator *S) {                  \
    TRY_TO(WalkUpFromUnary##NAME(S));                           \
    TRY_TO(TraverseStmt(S->getSubExpr()));                      \
    return true;                                                \
  }                                                             \
  bool WalkUpFromUnary##NAME(UnaryOperator *S) {                \
    TRY_TO(WalkUpFromUnaryOperator(S));                         \
    TRY_TO(VisitUnary##NAME(S));                                \
    return true;                                                \
  }                                                             \
  bool VisitUnary##NAME(UnaryOperator *S) { return true; }

  UNARYOP_LIST()
#undef OPERATOR

  // Define Traverse*(), WalkUpFrom*(), and Visit*() for binary
  // operator methods.  Binary operators are not classes in themselves
  // (they're all opcodes in BinaryOperator) but do have visitors.
#define GENERAL_BINOP_FALLBACK(NAME, BINOP_TYPE)                \
  bool TraverseBin##NAME(BINOP_TYPE *S) {                       \
    TRY_TO(WalkUpFromBin##NAME(S));                             \
    TRY_TO(TraverseStmt(S->getLHS()));                          \
    TRY_TO(TraverseStmt(S->getRHS()));                          \
    return true;                                                \
  }                                                             \
  bool WalkUpFromBin##NAME(BINOP_TYPE *S) {                     \
    TRY_TO(WalkUpFrom##BINOP_TYPE(S));                          \
    TRY_TO(VisitBin##NAME(S));                                  \
    return true;                                                \
  }                                                             \
  bool VisitBin##NAME(BINOP_TYPE *S) { return true; }

#define OPERATOR(NAME) GENERAL_BINOP_FALLBACK(NAME, BinaryOperator)
  BINOP_LIST()
#undef OPERATOR

  // Define Traverse*(), WalkUpFrom*(), and Visit*() for compound
  // assignment methods.  Compound assignment operators are not
  // classes in themselves (they're all opcodes in
  // CompoundAssignOperator) but do have visitors.
#define OPERATOR(NAME) \
  GENERAL_BINOP_FALLBACK(NAME##Assign, CompoundAssignOperator)

  CAO_LIST()
#undef OPERATOR
#undef GENERAL_BINOP_FALLBACK

  // ---- Methods on Types ----
  // FIXME: revamp to take TypeLoc's rather than Types.

  // Declare Traverse*() for all concrete Type classes.
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE) \
  bool Traverse##CLASS##Type(CLASS##Type *T);
#include "clang/AST/TypeNodes.def"
  // The above header #undefs ABSTRACT_TYPE and TYPE upon exit.

  // Define WalkUpFrom*() and empty Visit*() for all Type classes.
  bool WalkUpFromType(Type *T) { return getDerived().VisitType(T); }
  bool VisitType(Type *T) { return true; }
#define TYPE(CLASS, BASE)                                       \
  bool WalkUpFrom##CLASS##Type(CLASS##Type *T) {                \
    TRY_TO(WalkUpFrom##BASE(T));                                \
    TRY_TO(Visit##CLASS##Type(T));                              \
    return true;                                                \
  }                                                             \
  bool Visit##CLASS##Type(CLASS##Type *T) { return true; }
#include "clang/AST/TypeNodes.def"

  // ---- Methods on TypeLocs ----
  // FIXME: this currently just calls the matching Type methods

  // Declare Traverse*() for all concrete Type classes.
#define ABSTRACT_TYPELOC(CLASS, BASE)
#define TYPELOC(CLASS, BASE) \
  bool Traverse##CLASS##TypeLoc(CLASS##TypeLoc TL);
#include "clang/AST/TypeLocNodes.def"
  // The above header #undefs ABSTRACT_TYPELOC and TYPELOC upon exit.

  // Define WalkUpFrom*() and empty Visit*() for all TypeLoc classes.
  bool WalkUpFromTypeLoc(TypeLoc TL) { return getDerived().VisitTypeLoc(TL); }
  bool VisitTypeLoc(TypeLoc TL) { return true; }

  // QualifiedTypeLoc and UnqualTypeLoc are not declared in
  // TypeNodes.def and thus need to be handled specially.
  bool WalkUpFromQualifiedTypeLoc(QualifiedTypeLoc TL) {
    return getDerived().VisitUnqualTypeLoc(TL.getUnqualifiedLoc());
  }
  bool VisitQualifiedTypeLoc(QualifiedTypeLoc TL) { return true; }
  bool WalkUpFromUnqualTypeLoc(UnqualTypeLoc TL) {
    return getDerived().VisitUnqualTypeLoc(TL.getUnqualifiedLoc());
  }
  bool VisitUnqualTypeLoc(UnqualTypeLoc TL) { return true; }

  // Note that BASE includes trailing 'Type' which CLASS doesn't.
#define TYPE(CLASS, BASE)                                       \
  bool WalkUpFrom##CLASS##TypeLoc(CLASS##TypeLoc TL) {          \
    TRY_TO(WalkUpFrom##BASE##Loc(TL));                          \
    TRY_TO(Visit##CLASS##TypeLoc(TL));                          \
    return true;                                                \
  }                                                             \
  bool Visit##CLASS##TypeLoc(CLASS##TypeLoc TL) { return true; }
#include "clang/AST/TypeNodes.def"

  // ---- Methods on Decls ----

  // Declare Traverse*() for all concrete Decl classes.
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE) \
  bool Traverse##CLASS##Decl(CLASS##Decl *D);
#include "clang/AST/DeclNodes.inc"
  // The above header #undefs ABSTRACT_DECL and DECL upon exit.

  // Define WalkUpFrom*() and empty Visit*() for all Decl classes.
  bool WalkUpFromDecl(Decl *D) { return getDerived().VisitDecl(D); }
  bool VisitDecl(Decl *D) { return true; }
#define DECL(CLASS, BASE)                                       \
  bool WalkUpFrom##CLASS##Decl(CLASS##Decl *D) {                \
    TRY_TO(WalkUpFrom##BASE(D));                                \
    TRY_TO(Visit##CLASS##Decl(D));                              \
    return true;                                                \
  }                                                             \
  bool Visit##CLASS##Decl(CLASS##Decl *D) { return true; }
#include "clang/AST/DeclNodes.inc"

private:
  // These are helper methods used by more than one Traverse* method.
  bool TraverseTemplateParameterListHelper(TemplateParameterList *TPL);
  bool TraverseClassInstantiations(ClassTemplateDecl *D);
  bool TraverseFunctionInstantiations(FunctionTemplateDecl *D) ;
  bool TraverseTemplateArgumentLocsHelper(const TemplateArgumentLoc *TAL,
                                          unsigned Count);
  bool TraverseArrayTypeLocHelper(ArrayTypeLoc TL);
  bool TraverseRecordHelper(RecordDecl *D);
  bool TraverseCXXRecordHelper(CXXRecordDecl *D);
  bool TraverseDeclaratorHelper(DeclaratorDecl *D);
  bool TraverseDeclContextHelper(DeclContext *DC);
  bool TraverseFunctionHelper(FunctionDecl *D);
  bool TraverseVarHelper(VarDecl *D);

  struct EnqueueJob {
    Stmt *S;
    Stmt::child_iterator StmtIt;

    EnqueueJob(Stmt *S) : S(S), StmtIt() {}
  };
  bool dataTraverse(Stmt *S);
  bool dataTraverseNode(Stmt *S, bool &EnqueueChildren);
};

template<typename Derived>
bool RecursiveASTVisitor<Derived>::dataTraverse(Stmt *S) {

  SmallVector<EnqueueJob, 16> Queue;
  Queue.push_back(S);

  while (!Queue.empty()) {
    EnqueueJob &job = Queue.back();
    Stmt *CurrS = job.S;
    if (!CurrS) {
      Queue.pop_back();
      continue;
    }

    if (getDerived().shouldUseDataRecursionFor(CurrS)) {
      if (job.StmtIt == Stmt::child_iterator()) {
        bool EnqueueChildren = true;
        if (!dataTraverseNode(CurrS, EnqueueChildren)) return false;
        if (!EnqueueChildren) {
          Queue.pop_back();
          continue;
        }
        job.StmtIt = CurrS->child_begin();
      } else {
        ++job.StmtIt;
      }

      if (job.StmtIt != CurrS->child_end())
        Queue.push_back(*job.StmtIt);
      else
        Queue.pop_back();
      continue;
    }

    Queue.pop_back();
    TRY_TO(TraverseStmt(CurrS));
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::dataTraverseNode(Stmt *S,
                                                    bool &EnqueueChildren) {

// The cast for DISPATCH_WALK is needed for older versions of g++, but causes
// problems for MSVC.  So we'll skip the cast entirely for MSVC.
#if defined(_MSC_VER)
  #define GCC_CAST(CLASS)
#else
  #define GCC_CAST(CLASS) (bool (RecursiveASTVisitor::*)(CLASS*))
#endif

  // Dispatch to the corresponding WalkUpFrom* function only if the derived
  // class didn't override Traverse* (and thus the traversal is trivial).
#define DISPATCH_WALK(NAME, CLASS, VAR) \
  if (&RecursiveASTVisitor::Traverse##NAME == \
      GCC_CAST(CLASS)&Derived::Traverse##NAME) \
    return getDerived().WalkUpFrom##NAME(static_cast<CLASS*>(VAR)); \
  EnqueueChildren = false; \
  return getDerived().Traverse##NAME(static_cast<CLASS*>(VAR));

  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(S)) {
    switch (BinOp->getOpcode()) {
#define OPERATOR(NAME) \
    case BO_##NAME: DISPATCH_WALK(Bin##NAME, BinaryOperator, S);

    BINOP_LIST()
#undef OPERATOR

#define OPERATOR(NAME)                                          \
    case BO_##NAME##Assign:                          \
    DISPATCH_WALK(Bin##NAME##Assign, CompoundAssignOperator, S);

    CAO_LIST()
#undef OPERATOR
    }
  } else if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(S)) {
    switch (UnOp->getOpcode()) {
#define OPERATOR(NAME)                                                  \
    case UO_##NAME: DISPATCH_WALK(Unary##NAME, UnaryOperator, S);

    UNARYOP_LIST()
#undef OPERATOR
    }
  }

  // Top switch stmt: dispatch to TraverseFooStmt for each concrete FooStmt.
  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass: break;
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT) \
  case Stmt::CLASS##Class: DISPATCH_WALK(CLASS, CLASS, S);
#include "clang/AST/StmtNodes.inc"
  }

#undef DISPATCH_WALK
#undef GCC_CAST

  return true;
}

#define DISPATCH(NAME, CLASS, VAR) \
  return getDerived().Traverse##NAME(static_cast<CLASS*>(VAR))

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseStmt(Stmt *S) {
  if (!S)
    return true;

  if (getDerived().shouldUseDataRecursionFor(S))
    return dataTraverse(S);

  // If we have a binary expr, dispatch to the subcode of the binop.  A smart
  // optimizer (e.g. LLVM) will fold this comparison into the switch stmt
  // below.
  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(S)) {
    switch (BinOp->getOpcode()) {
#define OPERATOR(NAME) \
    case BO_##NAME: DISPATCH(Bin##NAME, BinaryOperator, S);

    BINOP_LIST()
#undef OPERATOR
#undef BINOP_LIST

#define OPERATOR(NAME)                                          \
    case BO_##NAME##Assign:                          \
      DISPATCH(Bin##NAME##Assign, CompoundAssignOperator, S);

    CAO_LIST()
#undef OPERATOR
#undef CAO_LIST
    }
  } else if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(S)) {
    switch (UnOp->getOpcode()) {
#define OPERATOR(NAME)                                                  \
    case UO_##NAME: DISPATCH(Unary##NAME, UnaryOperator, S);

    UNARYOP_LIST()
#undef OPERATOR
#undef UNARYOP_LIST
    }
  }

  // Top switch stmt: dispatch to TraverseFooStmt for each concrete FooStmt.
  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass: break;
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT) \
  case Stmt::CLASS##Class: DISPATCH(CLASS, CLASS, S);
#include "clang/AST/StmtNodes.inc"
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseType(QualType T) {
  if (T.isNull())
    return true;

  switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE) \
  case Type::CLASS: DISPATCH(CLASS##Type, CLASS##Type, \
                             const_cast<Type*>(T.getTypePtr()));
#include "clang/AST/TypeNodes.def"
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTypeLoc(TypeLoc TL) {
  if (TL.isNull())
    return true;

  switch (TL.getTypeLocClass()) {
#define ABSTRACT_TYPELOC(CLASS, BASE)
#define TYPELOC(CLASS, BASE) \
  case TypeLoc::CLASS: \
    return getDerived().Traverse##CLASS##TypeLoc(*cast<CLASS##TypeLoc>(&TL));
#include "clang/AST/TypeLocNodes.def"
  }

  return true;
}


template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseDecl(Decl *D) {
  if (!D)
    return true;

  // As a syntax visitor, by default we want to ignore declarations for
  // implicit declarations (ones not typed explicitly by the user).
  if (!getDerived().shouldVisitImplicitCode() && D->isImplicit())
    return true;

  switch (D->getKind()) {
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE) \
  case Decl::CLASS: DISPATCH(CLASS##Decl, CLASS##Decl, D);
#include "clang/AST/DeclNodes.inc"
 }

  return true;
}

#undef DISPATCH

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseNestedNameSpecifier(
                                                    NestedNameSpecifier *NNS) {
  if (!NNS)
    return true;

  if (NNS->getPrefix())
    TRY_TO(TraverseNestedNameSpecifier(NNS->getPrefix()));

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
  case NestedNameSpecifier::Namespace:
  case NestedNameSpecifier::NamespaceAlias:
  case NestedNameSpecifier::Global:
    return true;

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
    TRY_TO(TraverseType(QualType(NNS->getAsType(), 0)));
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseNestedNameSpecifierLoc(
                                                  NestedNameSpecifierLoc NNS) {
  if (!NNS)
    return true;

   if (NestedNameSpecifierLoc Prefix = NNS.getPrefix())
     TRY_TO(TraverseNestedNameSpecifierLoc(Prefix));

  switch (NNS.getNestedNameSpecifier()->getKind()) {
  case NestedNameSpecifier::Identifier:
  case NestedNameSpecifier::Namespace:
  case NestedNameSpecifier::NamespaceAlias:
  case NestedNameSpecifier::Global:
    return true;

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
    TRY_TO(TraverseTypeLoc(NNS.getTypeLoc()));
    break;
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseDeclarationNameInfo(
                                                 DeclarationNameInfo NameInfo) {
  switch (NameInfo.getName().getNameKind()) {
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    if (TypeSourceInfo *TSInfo = NameInfo.getNamedTypeInfo())
      TRY_TO(TraverseTypeLoc(TSInfo->getTypeLoc()));

    break;

  case DeclarationName::Identifier:
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::CXXOperatorName:
  case DeclarationName::CXXLiteralOperatorName:
  case DeclarationName::CXXUsingDirective:
    break;
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateName(TemplateName Template) {
  if (DependentTemplateName *DTN = Template.getAsDependentTemplateName())
    TRY_TO(TraverseNestedNameSpecifier(DTN->getQualifier()));
  else if (QualifiedTemplateName *QTN = Template.getAsQualifiedTemplateName())
    TRY_TO(TraverseNestedNameSpecifier(QTN->getQualifier()));

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateArgument(
                                                const TemplateArgument &Arg) {
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
  case TemplateArgument::Declaration:
  case TemplateArgument::Integral:
  case TemplateArgument::NullPtr:
    return true;

  case TemplateArgument::Type:
    return getDerived().TraverseType(Arg.getAsType());

  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
    return getDerived().TraverseTemplateName(
                                          Arg.getAsTemplateOrTemplatePattern());

  case TemplateArgument::Expression:
    return getDerived().TraverseStmt(Arg.getAsExpr());

  case TemplateArgument::Pack:
    return getDerived().TraverseTemplateArguments(Arg.pack_begin(),
                                                  Arg.pack_size());
  }

  return true;
}

// FIXME: no template name location?
// FIXME: no source locations for a template argument pack?
template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateArgumentLoc(
                                           const TemplateArgumentLoc &ArgLoc) {
  const TemplateArgument &Arg = ArgLoc.getArgument();

  switch (Arg.getKind()) {
  case TemplateArgument::Null:
  case TemplateArgument::Declaration:
  case TemplateArgument::Integral:
  case TemplateArgument::NullPtr:
    return true;

  case TemplateArgument::Type: {
    // FIXME: how can TSI ever be NULL?
    if (TypeSourceInfo *TSI = ArgLoc.getTypeSourceInfo())
      return getDerived().TraverseTypeLoc(TSI->getTypeLoc());
    else
      return getDerived().TraverseType(Arg.getAsType());
  }

  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
    if (ArgLoc.getTemplateQualifierLoc())
      TRY_TO(getDerived().TraverseNestedNameSpecifierLoc(
                                            ArgLoc.getTemplateQualifierLoc()));
    return getDerived().TraverseTemplateName(
                                         Arg.getAsTemplateOrTemplatePattern());

  case TemplateArgument::Expression:
    return getDerived().TraverseStmt(ArgLoc.getSourceExpression());

  case TemplateArgument::Pack:
    return getDerived().TraverseTemplateArguments(Arg.pack_begin(),
                                                  Arg.pack_size());
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateArguments(
                                                  const TemplateArgument *Args,
                                                            unsigned NumArgs) {
  for (unsigned I = 0; I != NumArgs; ++I) {
    TRY_TO(TraverseTemplateArgument(Args[I]));
  }

  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseConstructorInitializer(
                                                     CXXCtorInitializer *Init) {
  if (TypeSourceInfo *TInfo = Init->getTypeSourceInfo())
    TRY_TO(TraverseTypeLoc(TInfo->getTypeLoc()));

  if (Init->isWritten() || getDerived().shouldVisitImplicitCode())
    TRY_TO(TraverseStmt(Init->getInit()));
  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseLambdaCapture(LambdaExpr::Capture C){
  return true;
}

// ----------------- Type traversal -----------------

// This macro makes available a variable T, the passed-in type.
#define DEF_TRAVERSE_TYPE(TYPE, CODE)                     \
  template<typename Derived>                                           \
  bool RecursiveASTVisitor<Derived>::Traverse##TYPE (TYPE *T) {        \
    TRY_TO(WalkUpFrom##TYPE (T));                                      \
    { CODE; }                                                          \
    return true;                                                       \
  }

DEF_TRAVERSE_TYPE(BuiltinType, { })

DEF_TRAVERSE_TYPE(ComplexType, {
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(PointerType, {
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(BlockPointerType, {
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(LValueReferenceType, {
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(RValueReferenceType, {
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(MemberPointerType, {
    TRY_TO(TraverseType(QualType(T->getClass(), 0)));
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(ConstantArrayType, {
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(IncompleteArrayType, {
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(VariableArrayType, {
    TRY_TO(TraverseType(T->getElementType()));
    TRY_TO(TraverseStmt(T->getSizeExpr()));
  })

DEF_TRAVERSE_TYPE(DependentSizedArrayType, {
    TRY_TO(TraverseType(T->getElementType()));
    if (T->getSizeExpr())
      TRY_TO(TraverseStmt(T->getSizeExpr()));
  })

DEF_TRAVERSE_TYPE(DependentSizedExtVectorType, {
    if (T->getSizeExpr())
      TRY_TO(TraverseStmt(T->getSizeExpr()));
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(VectorType, {
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(ExtVectorType, {
    TRY_TO(TraverseType(T->getElementType()));
  })

DEF_TRAVERSE_TYPE(FunctionNoProtoType, {
    TRY_TO(TraverseType(T->getResultType()));
  })

DEF_TRAVERSE_TYPE(FunctionProtoType, {
    TRY_TO(TraverseType(T->getResultType()));

    for (FunctionProtoType::arg_type_iterator A = T->arg_type_begin(),
                                           AEnd = T->arg_type_end();
         A != AEnd; ++A) {
      TRY_TO(TraverseType(*A));
    }

    for (FunctionProtoType::exception_iterator E = T->exception_begin(),
                                            EEnd = T->exception_end();
         E != EEnd; ++E) {
      TRY_TO(TraverseType(*E));
    }
  })

DEF_TRAVERSE_TYPE(UnresolvedUsingType, { })
DEF_TRAVERSE_TYPE(TypedefType, { })

DEF_TRAVERSE_TYPE(TypeOfExprType, {
    TRY_TO(TraverseStmt(T->getUnderlyingExpr()));
  })

DEF_TRAVERSE_TYPE(TypeOfType, {
    TRY_TO(TraverseType(T->getUnderlyingType()));
  })

DEF_TRAVERSE_TYPE(DecltypeType, {
    TRY_TO(TraverseStmt(T->getUnderlyingExpr()));
  })

DEF_TRAVERSE_TYPE(UnaryTransformType, {
    TRY_TO(TraverseType(T->getBaseType()));
    TRY_TO(TraverseType(T->getUnderlyingType()));
    })

DEF_TRAVERSE_TYPE(AutoType, {
    TRY_TO(TraverseType(T->getDeducedType()));
  })

DEF_TRAVERSE_TYPE(RecordType, { })
DEF_TRAVERSE_TYPE(EnumType, { })
DEF_TRAVERSE_TYPE(TemplateTypeParmType, { })
DEF_TRAVERSE_TYPE(SubstTemplateTypeParmType, { })
DEF_TRAVERSE_TYPE(SubstTemplateTypeParmPackType, { })

DEF_TRAVERSE_TYPE(TemplateSpecializationType, {
    TRY_TO(TraverseTemplateName(T->getTemplateName()));
    TRY_TO(TraverseTemplateArguments(T->getArgs(), T->getNumArgs()));
  })

DEF_TRAVERSE_TYPE(InjectedClassNameType, { })

DEF_TRAVERSE_TYPE(AttributedType, {
    TRY_TO(TraverseType(T->getModifiedType()));
  })

DEF_TRAVERSE_TYPE(ParenType, {
    TRY_TO(TraverseType(T->getInnerType()));
  })

DEF_TRAVERSE_TYPE(ElaboratedType, {
    if (T->getQualifier()) {
      TRY_TO(TraverseNestedNameSpecifier(T->getQualifier()));
    }
    TRY_TO(TraverseType(T->getNamedType()));
  })

DEF_TRAVERSE_TYPE(DependentNameType, {
    TRY_TO(TraverseNestedNameSpecifier(T->getQualifier()));
  })

DEF_TRAVERSE_TYPE(DependentTemplateSpecializationType, {
    TRY_TO(TraverseNestedNameSpecifier(T->getQualifier()));
    TRY_TO(TraverseTemplateArguments(T->getArgs(), T->getNumArgs()));
  })

DEF_TRAVERSE_TYPE(PackExpansionType, {
    TRY_TO(TraverseType(T->getPattern()));
  })

DEF_TRAVERSE_TYPE(ObjCInterfaceType, { })

DEF_TRAVERSE_TYPE(ObjCObjectType, {
    // We have to watch out here because an ObjCInterfaceType's base
    // type is itself.
    if (T->getBaseType().getTypePtr() != T)
      TRY_TO(TraverseType(T->getBaseType()));
  })

DEF_TRAVERSE_TYPE(ObjCObjectPointerType, {
    TRY_TO(TraverseType(T->getPointeeType()));
  })

DEF_TRAVERSE_TYPE(AtomicType, {
    TRY_TO(TraverseType(T->getValueType()));
  })

#undef DEF_TRAVERSE_TYPE

// ----------------- TypeLoc traversal -----------------

// This macro makes available a variable TL, the passed-in TypeLoc.
// If requested, it calls WalkUpFrom* for the Type in the given TypeLoc,
// in addition to WalkUpFrom* for the TypeLoc itself, such that existing
// clients that override the WalkUpFrom*Type() and/or Visit*Type() methods
// continue to work.
#define DEF_TRAVERSE_TYPELOC(TYPE, CODE)                                \
  template<typename Derived>                                            \
  bool RecursiveASTVisitor<Derived>::Traverse##TYPE##Loc(TYPE##Loc TL) { \
    if (getDerived().shouldWalkTypesOfTypeLocs())                       \
      TRY_TO(WalkUpFrom##TYPE(const_cast<TYPE*>(TL.getTypePtr())));     \
    TRY_TO(WalkUpFrom##TYPE##Loc(TL));                                  \
    { CODE; }                                                           \
    return true;                                                        \
  }

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseQualifiedTypeLoc(
    QualifiedTypeLoc TL) {
  // Move this over to the 'main' typeloc tree.  Note that this is a
  // move -- we pretend that we were really looking at the unqualified
  // typeloc all along -- rather than a recursion, so we don't follow
  // the normal CRTP plan of going through
  // getDerived().TraverseTypeLoc.  If we did, we'd be traversing
  // twice for the same type (once as a QualifiedTypeLoc version of
  // the type, once as an UnqualifiedTypeLoc version of the type),
  // which in effect means we'd call VisitTypeLoc twice with the
  // 'same' type.  This solves that problem, at the cost of never
  // seeing the qualified version of the type (unless the client
  // subclasses TraverseQualifiedTypeLoc themselves).  It's not a
  // perfect solution.  A perfect solution probably requires making
  // QualifiedTypeLoc a wrapper around TypeLoc -- like QualType is a
  // wrapper around Type* -- rather than being its own class in the
  // type hierarchy.
  return TraverseTypeLoc(TL.getUnqualifiedLoc());
}

DEF_TRAVERSE_TYPELOC(BuiltinType, { })

// FIXME: ComplexTypeLoc is unfinished
DEF_TRAVERSE_TYPELOC(ComplexType, {
    TRY_TO(TraverseType(TL.getTypePtr()->getElementType()));
  })

DEF_TRAVERSE_TYPELOC(PointerType, {
    TRY_TO(TraverseTypeLoc(TL.getPointeeLoc()));
  })

DEF_TRAVERSE_TYPELOC(BlockPointerType, {
    TRY_TO(TraverseTypeLoc(TL.getPointeeLoc()));
  })

DEF_TRAVERSE_TYPELOC(LValueReferenceType, {
    TRY_TO(TraverseTypeLoc(TL.getPointeeLoc()));
  })

DEF_TRAVERSE_TYPELOC(RValueReferenceType, {
    TRY_TO(TraverseTypeLoc(TL.getPointeeLoc()));
  })

// FIXME: location of base class?
// We traverse this in the type case as well, but how is it not reached through
// the pointee type?
DEF_TRAVERSE_TYPELOC(MemberPointerType, {
    TRY_TO(TraverseType(QualType(TL.getTypePtr()->getClass(), 0)));
    TRY_TO(TraverseTypeLoc(TL.getPointeeLoc()));
  })

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseArrayTypeLocHelper(ArrayTypeLoc TL) {
  // This isn't available for ArrayType, but is for the ArrayTypeLoc.
  TRY_TO(TraverseStmt(TL.getSizeExpr()));
  return true;
}

DEF_TRAVERSE_TYPELOC(ConstantArrayType, {
    TRY_TO(TraverseTypeLoc(TL.getElementLoc()));
    return TraverseArrayTypeLocHelper(TL);
  })

DEF_TRAVERSE_TYPELOC(IncompleteArrayType, {
    TRY_TO(TraverseTypeLoc(TL.getElementLoc()));
    return TraverseArrayTypeLocHelper(TL);
  })

DEF_TRAVERSE_TYPELOC(VariableArrayType, {
    TRY_TO(TraverseTypeLoc(TL.getElementLoc()));
    return TraverseArrayTypeLocHelper(TL);
  })

DEF_TRAVERSE_TYPELOC(DependentSizedArrayType, {
    TRY_TO(TraverseTypeLoc(TL.getElementLoc()));
    return TraverseArrayTypeLocHelper(TL);
  })

// FIXME: order? why not size expr first?
// FIXME: base VectorTypeLoc is unfinished
DEF_TRAVERSE_TYPELOC(DependentSizedExtVectorType, {
    if (TL.getTypePtr()->getSizeExpr())
      TRY_TO(TraverseStmt(TL.getTypePtr()->getSizeExpr()));
    TRY_TO(TraverseType(TL.getTypePtr()->getElementType()));
  })

// FIXME: VectorTypeLoc is unfinished
DEF_TRAVERSE_TYPELOC(VectorType, {
    TRY_TO(TraverseType(TL.getTypePtr()->getElementType()));
  })

// FIXME: size and attributes
// FIXME: base VectorTypeLoc is unfinished
DEF_TRAVERSE_TYPELOC(ExtVectorType, {
    TRY_TO(TraverseType(TL.getTypePtr()->getElementType()));
  })

DEF_TRAVERSE_TYPELOC(FunctionNoProtoType, {
    TRY_TO(TraverseTypeLoc(TL.getResultLoc()));
  })

// FIXME: location of exception specifications (attributes?)
DEF_TRAVERSE_TYPELOC(FunctionProtoType, {
    TRY_TO(TraverseTypeLoc(TL.getResultLoc()));

    const FunctionProtoType *T = TL.getTypePtr();

    for (unsigned I = 0, E = TL.getNumArgs(); I != E; ++I) {
      if (TL.getArg(I)) {
        TRY_TO(TraverseDecl(TL.getArg(I)));
      } else if (I < T->getNumArgs()) {
        TRY_TO(TraverseType(T->getArgType(I)));
      }
    }

    for (FunctionProtoType::exception_iterator E = T->exception_begin(),
                                            EEnd = T->exception_end();
         E != EEnd; ++E) {
      TRY_TO(TraverseType(*E));
    }
  })

DEF_TRAVERSE_TYPELOC(UnresolvedUsingType, { })
DEF_TRAVERSE_TYPELOC(TypedefType, { })

DEF_TRAVERSE_TYPELOC(TypeOfExprType, {
    TRY_TO(TraverseStmt(TL.getUnderlyingExpr()));
  })

DEF_TRAVERSE_TYPELOC(TypeOfType, {
    TRY_TO(TraverseTypeLoc(TL.getUnderlyingTInfo()->getTypeLoc()));
  })

// FIXME: location of underlying expr
DEF_TRAVERSE_TYPELOC(DecltypeType, {
    TRY_TO(TraverseStmt(TL.getTypePtr()->getUnderlyingExpr()));
  })

DEF_TRAVERSE_TYPELOC(UnaryTransformType, {
    TRY_TO(TraverseTypeLoc(TL.getUnderlyingTInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_TYPELOC(AutoType, {
    TRY_TO(TraverseType(TL.getTypePtr()->getDeducedType()));
  })

DEF_TRAVERSE_TYPELOC(RecordType, { })
DEF_TRAVERSE_TYPELOC(EnumType, { })
DEF_TRAVERSE_TYPELOC(TemplateTypeParmType, { })
DEF_TRAVERSE_TYPELOC(SubstTemplateTypeParmType, { })
DEF_TRAVERSE_TYPELOC(SubstTemplateTypeParmPackType, { })

// FIXME: use the loc for the template name?
DEF_TRAVERSE_TYPELOC(TemplateSpecializationType, {
    TRY_TO(TraverseTemplateName(TL.getTypePtr()->getTemplateName()));
    for (unsigned I = 0, E = TL.getNumArgs(); I != E; ++I) {
      TRY_TO(TraverseTemplateArgumentLoc(TL.getArgLoc(I)));
    }
  })

DEF_TRAVERSE_TYPELOC(InjectedClassNameType, { })

DEF_TRAVERSE_TYPELOC(ParenType, {
    TRY_TO(TraverseTypeLoc(TL.getInnerLoc()));
  })

DEF_TRAVERSE_TYPELOC(AttributedType, {
    TRY_TO(TraverseTypeLoc(TL.getModifiedLoc()));
  })

DEF_TRAVERSE_TYPELOC(ElaboratedType, {
    if (TL.getQualifierLoc()) {
      TRY_TO(TraverseNestedNameSpecifierLoc(TL.getQualifierLoc()));
    }
    TRY_TO(TraverseTypeLoc(TL.getNamedTypeLoc()));
  })

DEF_TRAVERSE_TYPELOC(DependentNameType, {
    TRY_TO(TraverseNestedNameSpecifierLoc(TL.getQualifierLoc()));
  })

DEF_TRAVERSE_TYPELOC(DependentTemplateSpecializationType, {
    if (TL.getQualifierLoc()) {
      TRY_TO(TraverseNestedNameSpecifierLoc(TL.getQualifierLoc()));
    }

    for (unsigned I = 0, E = TL.getNumArgs(); I != E; ++I) {
      TRY_TO(TraverseTemplateArgumentLoc(TL.getArgLoc(I)));
    }
  })

DEF_TRAVERSE_TYPELOC(PackExpansionType, {
    TRY_TO(TraverseTypeLoc(TL.getPatternLoc()));
  })

DEF_TRAVERSE_TYPELOC(ObjCInterfaceType, { })

DEF_TRAVERSE_TYPELOC(ObjCObjectType, {
    // We have to watch out here because an ObjCInterfaceType's base
    // type is itself.
    if (TL.getTypePtr()->getBaseType().getTypePtr() != TL.getTypePtr())
      TRY_TO(TraverseTypeLoc(TL.getBaseLoc()));
  })

DEF_TRAVERSE_TYPELOC(ObjCObjectPointerType, {
    TRY_TO(TraverseTypeLoc(TL.getPointeeLoc()));
  })

DEF_TRAVERSE_TYPELOC(AtomicType, {
    TRY_TO(TraverseTypeLoc(TL.getValueLoc()));
  })

#undef DEF_TRAVERSE_TYPELOC

// ----------------- Decl traversal -----------------
//
// For a Decl, we automate (in the DEF_TRAVERSE_DECL macro) traversing
// the children that come from the DeclContext associated with it.
// Therefore each Traverse* only needs to worry about children other
// than those.

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseDeclContextHelper(DeclContext *DC) {
  if (!DC)
    return true;

  for (DeclContext::decl_iterator Child = DC->decls_begin(),
           ChildEnd = DC->decls_end();
       Child != ChildEnd; ++Child) {
    // BlockDecls are traversed through BlockExprs.
    if (!isa<BlockDecl>(*Child))
      TRY_TO(TraverseDecl(*Child));
  }

  return true;
}

// This macro makes available a variable D, the passed-in decl.
#define DEF_TRAVERSE_DECL(DECL, CODE)                           \
template<typename Derived>                                      \
bool RecursiveASTVisitor<Derived>::Traverse##DECL (DECL *D) {   \
  TRY_TO(WalkUpFrom##DECL (D));                                 \
  { CODE; }                                                     \
  TRY_TO(TraverseDeclContextHelper(dyn_cast<DeclContext>(D)));  \
  return true;                                                  \
}

DEF_TRAVERSE_DECL(AccessSpecDecl, { })

DEF_TRAVERSE_DECL(BlockDecl, {
    if (TypeSourceInfo *TInfo = D->getSignatureAsWritten())
      TRY_TO(TraverseTypeLoc(TInfo->getTypeLoc()));
    TRY_TO(TraverseStmt(D->getBody()));
    // This return statement makes sure the traversal of nodes in
    // decls_begin()/decls_end() (done in the DEF_TRAVERSE_DECL macro)
    // is skipped - don't remove it.
    return true;
  })

DEF_TRAVERSE_DECL(FileScopeAsmDecl, {
    TRY_TO(TraverseStmt(D->getAsmString()));
  })

DEF_TRAVERSE_DECL(ImportDecl, { })

DEF_TRAVERSE_DECL(FriendDecl, {
    // Friend is either decl or a type.
    if (D->getFriendType())
      TRY_TO(TraverseTypeLoc(D->getFriendType()->getTypeLoc()));
    else
      TRY_TO(TraverseDecl(D->getFriendDecl()));
  })

DEF_TRAVERSE_DECL(FriendTemplateDecl, {
    if (D->getFriendType())
      TRY_TO(TraverseTypeLoc(D->getFriendType()->getTypeLoc()));
    else
      TRY_TO(TraverseDecl(D->getFriendDecl()));
    for (unsigned I = 0, E = D->getNumTemplateParameters(); I < E; ++I) {
      TemplateParameterList *TPL = D->getTemplateParameterList(I);
      for (TemplateParameterList::iterator ITPL = TPL->begin(),
                                           ETPL = TPL->end();
           ITPL != ETPL; ++ITPL) {
        TRY_TO(TraverseDecl(*ITPL));
      }
    }
  })

DEF_TRAVERSE_DECL(ClassScopeFunctionSpecializationDecl, {
    TRY_TO(TraverseDecl(D->getSpecialization()));

    if (D->hasExplicitTemplateArgs()) {
      const TemplateArgumentListInfo& args = D->templateArgs();
      TRY_TO(TraverseTemplateArgumentLocsHelper(
          args.getArgumentArray(), args.size()));
    }
 })

DEF_TRAVERSE_DECL(LinkageSpecDecl, { })

DEF_TRAVERSE_DECL(ObjCPropertyImplDecl, {
    // FIXME: implement this
  })

DEF_TRAVERSE_DECL(StaticAssertDecl, {
    TRY_TO(TraverseStmt(D->getAssertExpr()));
    TRY_TO(TraverseStmt(D->getMessage()));
  })

DEF_TRAVERSE_DECL(TranslationUnitDecl, {
    // Code in an unnamed namespace shows up automatically in
    // decls_begin()/decls_end().  Thus we don't need to recurse on
    // D->getAnonymousNamespace().
  })

DEF_TRAVERSE_DECL(NamespaceAliasDecl, {
    // We shouldn't traverse an aliased namespace, since it will be
    // defined (and, therefore, traversed) somewhere else.
    //
    // This return statement makes sure the traversal of nodes in
    // decls_begin()/decls_end() (done in the DEF_TRAVERSE_DECL macro)
    // is skipped - don't remove it.
    return true;
  })

DEF_TRAVERSE_DECL(LabelDecl, {
  // There is no code in a LabelDecl.
})


DEF_TRAVERSE_DECL(NamespaceDecl, {
    // Code in an unnamed namespace shows up automatically in
    // decls_begin()/decls_end().  Thus we don't need to recurse on
    // D->getAnonymousNamespace().
  })

DEF_TRAVERSE_DECL(ObjCCompatibleAliasDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCCategoryDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCCategoryImplDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCImplementationDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCInterfaceDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCProtocolDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(ObjCMethodDecl, {
    if (D->getResultTypeSourceInfo()) {
      TRY_TO(TraverseTypeLoc(D->getResultTypeSourceInfo()->getTypeLoc()));
    }
    for (ObjCMethodDecl::param_iterator
           I = D->param_begin(), E = D->param_end(); I != E; ++I) {
      TRY_TO(TraverseDecl(*I));
    }
    if (D->isThisDeclarationADefinition()) {
      TRY_TO(TraverseStmt(D->getBody()));
    }
    return true;
  })

DEF_TRAVERSE_DECL(ObjCPropertyDecl, {
    // FIXME: implement
  })

DEF_TRAVERSE_DECL(UsingDecl, {
    TRY_TO(TraverseNestedNameSpecifierLoc(D->getQualifierLoc()));
    TRY_TO(TraverseDeclarationNameInfo(D->getNameInfo()));
  })

DEF_TRAVERSE_DECL(UsingDirectiveDecl, {
    TRY_TO(TraverseNestedNameSpecifierLoc(D->getQualifierLoc()));
  })

DEF_TRAVERSE_DECL(UsingShadowDecl, { })

// A helper method for TemplateDecl's children.
template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateParameterListHelper(
    TemplateParameterList *TPL) {
  if (TPL) {
    for (TemplateParameterList::iterator I = TPL->begin(), E = TPL->end();
         I != E; ++I) {
      TRY_TO(TraverseDecl(*I));
    }
  }
  return true;
}

// A helper method for traversing the implicit instantiations of a
// class template.
template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseClassInstantiations(
    ClassTemplateDecl *D) {
  ClassTemplateDecl::spec_iterator end = D->spec_end();
  for (ClassTemplateDecl::spec_iterator it = D->spec_begin(); it != end; ++it) {
    ClassTemplateSpecializationDecl* SD = *it;

    switch (SD->getSpecializationKind()) {
    // Visit the implicit instantiations with the requested pattern.
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
      TRY_TO(TraverseDecl(SD));
      break;

    // We don't need to do anything on an explicit instantiation
    // or explicit specialization because there will be an explicit
    // node for it elsewhere.
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
    case TSK_ExplicitSpecialization:
      break;
    }
  }

  return true;
}

DEF_TRAVERSE_DECL(ClassTemplateDecl, {
    CXXRecordDecl* TempDecl = D->getTemplatedDecl();
    TRY_TO(TraverseDecl(TempDecl));
    TRY_TO(TraverseTemplateParameterListHelper(D->getTemplateParameters()));

    // By default, we do not traverse the instantiations of
    // class templates since they do not appear in the user code. The
    // following code optionally traverses them.
    //
    // We only traverse the class instantiations when we see the canonical
    // declaration of the template, to ensure we only visit them once.
    if (getDerived().shouldVisitTemplateInstantiations() &&
        D == D->getCanonicalDecl())
      TRY_TO(TraverseClassInstantiations(D));

    // Note that getInstantiatedFromMemberTemplate() is just a link
    // from a template instantiation back to the template from which
    // it was instantiated, and thus should not be traversed.
  })

// A helper method for traversing the instantiations of a
// function while skipping its specializations.
template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseFunctionInstantiations(
    FunctionTemplateDecl *D) {
  FunctionTemplateDecl::spec_iterator end = D->spec_end();
  for (FunctionTemplateDecl::spec_iterator it = D->spec_begin(); it != end;
       ++it) {
    FunctionDecl* FD = *it;
    switch (FD->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
      // We don't know what kind of FunctionDecl this is.
      TRY_TO(TraverseDecl(FD));
      break;

    // FIXME: For now traverse explicit instantiations here. Change that
    // once they are represented as dedicated nodes in the AST.
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
      TRY_TO(TraverseDecl(FD));
      break;

    case TSK_ExplicitSpecialization:
      break;
    }
  }

  return true;
}

DEF_TRAVERSE_DECL(FunctionTemplateDecl, {
    TRY_TO(TraverseDecl(D->getTemplatedDecl()));
    TRY_TO(TraverseTemplateParameterListHelper(D->getTemplateParameters()));

    // By default, we do not traverse the instantiations of
    // function templates since they do not appear in the user code. The
    // following code optionally traverses them.
    //
    // We only traverse the function instantiations when we see the canonical
    // declaration of the template, to ensure we only visit them once.
    if (getDerived().shouldVisitTemplateInstantiations() &&
        D == D->getCanonicalDecl())
      TRY_TO(TraverseFunctionInstantiations(D));
  })

DEF_TRAVERSE_DECL(TemplateTemplateParmDecl, {
    // D is the "T" in something like
    //   template <template <typename> class T> class container { };
    TRY_TO(TraverseDecl(D->getTemplatedDecl()));
    if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited()) {
      TRY_TO(TraverseTemplateArgumentLoc(D->getDefaultArgument()));
    }
    TRY_TO(TraverseTemplateParameterListHelper(D->getTemplateParameters()));
  })

DEF_TRAVERSE_DECL(TemplateTypeParmDecl, {
    // D is the "T" in something like "template<typename T> class vector;"
    if (D->getTypeForDecl())
      TRY_TO(TraverseType(QualType(D->getTypeForDecl(), 0)));
    if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
      TRY_TO(TraverseTypeLoc(D->getDefaultArgumentInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_DECL(TypedefDecl, {
    TRY_TO(TraverseTypeLoc(D->getTypeSourceInfo()->getTypeLoc()));
    // We shouldn't traverse D->getTypeForDecl(); it's a result of
    // declaring the typedef, not something that was written in the
    // source.
  })

DEF_TRAVERSE_DECL(TypeAliasDecl, {
    TRY_TO(TraverseTypeLoc(D->getTypeSourceInfo()->getTypeLoc()));
    // We shouldn't traverse D->getTypeForDecl(); it's a result of
    // declaring the type alias, not something that was written in the
    // source.
  })

DEF_TRAVERSE_DECL(TypeAliasTemplateDecl, {
    TRY_TO(TraverseDecl(D->getTemplatedDecl()));
    TRY_TO(TraverseTemplateParameterListHelper(D->getTemplateParameters()));
  })

DEF_TRAVERSE_DECL(UnresolvedUsingTypenameDecl, {
    // A dependent using declaration which was marked with 'typename'.
    //   template<class T> class A : public B<T> { using typename B<T>::foo; };
    TRY_TO(TraverseNestedNameSpecifierLoc(D->getQualifierLoc()));
    // We shouldn't traverse D->getTypeForDecl(); it's a result of
    // declaring the type, not something that was written in the
    // source.
  })

DEF_TRAVERSE_DECL(EnumDecl, {
    if (D->getTypeForDecl())
      TRY_TO(TraverseType(QualType(D->getTypeForDecl(), 0)));

    TRY_TO(TraverseNestedNameSpecifierLoc(D->getQualifierLoc()));
    // The enumerators are already traversed by
    // decls_begin()/decls_end().
  })


// Helper methods for RecordDecl and its children.
template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseRecordHelper(
    RecordDecl *D) {
  // We shouldn't traverse D->getTypeForDecl(); it's a result of
  // declaring the type, not something that was written in the source.

  TRY_TO(TraverseNestedNameSpecifierLoc(D->getQualifierLoc()));
  return true;
}

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseCXXRecordHelper(
    CXXRecordDecl *D) {
  if (!TraverseRecordHelper(D))
    return false;
  if (D->isCompleteDefinition()) {
    for (CXXRecordDecl::base_class_iterator I = D->bases_begin(),
                                            E = D->bases_end();
         I != E; ++I) {
      TRY_TO(TraverseTypeLoc(I->getTypeSourceInfo()->getTypeLoc()));
    }
    // We don't traverse the friends or the conversions, as they are
    // already in decls_begin()/decls_end().
  }
  return true;
}

DEF_TRAVERSE_DECL(RecordDecl, {
    TRY_TO(TraverseRecordHelper(D));
  })

DEF_TRAVERSE_DECL(CXXRecordDecl, {
    TRY_TO(TraverseCXXRecordHelper(D));
  })

DEF_TRAVERSE_DECL(ClassTemplateSpecializationDecl, {
    // For implicit instantiations ("set<int> x;"), we don't want to
    // recurse at all, since the instatiated class isn't written in
    // the source code anywhere.  (Note the instatiated *type* --
    // set<int> -- is written, and will still get a callback of
    // TemplateSpecializationType).  For explicit instantiations
    // ("template set<int>;"), we do need a callback, since this
    // is the only callback that's made for this instantiation.
    // We use getTypeAsWritten() to distinguish.
    if (TypeSourceInfo *TSI = D->getTypeAsWritten())
      TRY_TO(TraverseTypeLoc(TSI->getTypeLoc()));

    if (!getDerived().shouldVisitTemplateInstantiations() &&
        D->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
      // Returning from here skips traversing the
      // declaration context of the ClassTemplateSpecializationDecl
      // (embedded in the DEF_TRAVERSE_DECL() macro)
      // which contains the instantiated members of the class.
      return true;
  })

template <typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseTemplateArgumentLocsHelper(
    const TemplateArgumentLoc *TAL, unsigned Count) {
  for (unsigned I = 0; I < Count; ++I) {
    TRY_TO(TraverseTemplateArgumentLoc(TAL[I]));
  }
  return true;
}

DEF_TRAVERSE_DECL(ClassTemplatePartialSpecializationDecl, {
    // The partial specialization.
    if (TemplateParameterList *TPL = D->getTemplateParameters()) {
      for (TemplateParameterList::iterator I = TPL->begin(), E = TPL->end();
           I != E; ++I) {
        TRY_TO(TraverseDecl(*I));
      }
    }
    // The args that remains unspecialized.
    TRY_TO(TraverseTemplateArgumentLocsHelper(
        D->getTemplateArgsAsWritten(), D->getNumTemplateArgsAsWritten()));

    // Don't need the ClassTemplatePartialSpecializationHelper, even
    // though that's our parent class -- we already visit all the
    // template args here.
    TRY_TO(TraverseCXXRecordHelper(D));

    // Instantiations will have been visited with the primary template.
  })

DEF_TRAVERSE_DECL(EnumConstantDecl, {
    TRY_TO(TraverseStmt(D->getInitExpr()));
  })

DEF_TRAVERSE_DECL(UnresolvedUsingValueDecl, {
    // Like UnresolvedUsingTypenameDecl, but without the 'typename':
    //    template <class T> Class A : public Base<T> { using Base<T>::foo; };
    TRY_TO(TraverseNestedNameSpecifierLoc(D->getQualifierLoc()));
    TRY_TO(TraverseDeclarationNameInfo(D->getNameInfo()));
  })

DEF_TRAVERSE_DECL(IndirectFieldDecl, {})

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseDeclaratorHelper(DeclaratorDecl *D) {
  TRY_TO(TraverseNestedNameSpecifierLoc(D->getQualifierLoc()));
  if (D->getTypeSourceInfo())
    TRY_TO(TraverseTypeLoc(D->getTypeSourceInfo()->getTypeLoc()));
  else
    TRY_TO(TraverseType(D->getType()));
  return true;
}

DEF_TRAVERSE_DECL(FieldDecl, {
    TRY_TO(TraverseDeclaratorHelper(D));
    if (D->isBitField())
      TRY_TO(TraverseStmt(D->getBitWidth()));
    else if (D->hasInClassInitializer())
      TRY_TO(TraverseStmt(D->getInClassInitializer()));
  })

DEF_TRAVERSE_DECL(ObjCAtDefsFieldDecl, {
    TRY_TO(TraverseDeclaratorHelper(D));
    if (D->isBitField())
      TRY_TO(TraverseStmt(D->getBitWidth()));
    // FIXME: implement the rest.
  })

DEF_TRAVERSE_DECL(ObjCIvarDecl, {
    TRY_TO(TraverseDeclaratorHelper(D));
    if (D->isBitField())
      TRY_TO(TraverseStmt(D->getBitWidth()));
    // FIXME: implement the rest.
  })

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseFunctionHelper(FunctionDecl *D) {
  TRY_TO(TraverseNestedNameSpecifierLoc(D->getQualifierLoc()));
  TRY_TO(TraverseDeclarationNameInfo(D->getNameInfo()));

  // If we're an explicit template specialization, iterate over the
  // template args that were explicitly specified.  If we were doing
  // this in typing order, we'd do it between the return type and
  // the function args, but both are handled by the FunctionTypeLoc
  // above, so we have to choose one side.  I've decided to do before.
  if (const FunctionTemplateSpecializationInfo *FTSI =
      D->getTemplateSpecializationInfo()) {
    if (FTSI->getTemplateSpecializationKind() != TSK_Undeclared &&
        FTSI->getTemplateSpecializationKind() != TSK_ImplicitInstantiation) {
      // A specialization might not have explicit template arguments if it has
      // a templated return type and concrete arguments.
      if (const ASTTemplateArgumentListInfo *TALI =
          FTSI->TemplateArgumentsAsWritten) {
        TRY_TO(TraverseTemplateArgumentLocsHelper(TALI->getTemplateArgs(),
                                                  TALI->NumTemplateArgs));
      }
    }
  }

  // Visit the function type itself, which can be either
  // FunctionNoProtoType or FunctionProtoType, or a typedef.  This
  // also covers the return type and the function parameters,
  // including exception specifications.
  if (clang::TypeSourceInfo *TSI = D->getTypeSourceInfo()) {
    TRY_TO(TraverseTypeLoc(TSI->getTypeLoc()));
  }

  if (CXXConstructorDecl *Ctor = dyn_cast<CXXConstructorDecl>(D)) {
    // Constructor initializers.
    for (CXXConstructorDecl::init_iterator I = Ctor->init_begin(),
                                           E = Ctor->init_end();
         I != E; ++I) {
      TRY_TO(TraverseConstructorInitializer(*I));
    }
  }

  if (D->isThisDeclarationADefinition()) {
    TRY_TO(TraverseStmt(D->getBody()));  // Function body.
  }
  return true;
}

DEF_TRAVERSE_DECL(FunctionDecl, {
    // We skip decls_begin/decls_end, which are already covered by
    // TraverseFunctionHelper().
    return TraverseFunctionHelper(D);
  })

DEF_TRAVERSE_DECL(CXXMethodDecl, {
    // We skip decls_begin/decls_end, which are already covered by
    // TraverseFunctionHelper().
    return TraverseFunctionHelper(D);
  })

DEF_TRAVERSE_DECL(CXXConstructorDecl, {
    // We skip decls_begin/decls_end, which are already covered by
    // TraverseFunctionHelper().
    return TraverseFunctionHelper(D);
  })

// CXXConversionDecl is the declaration of a type conversion operator.
// It's not a cast expression.
DEF_TRAVERSE_DECL(CXXConversionDecl, {
    // We skip decls_begin/decls_end, which are already covered by
    // TraverseFunctionHelper().
    return TraverseFunctionHelper(D);
  })

DEF_TRAVERSE_DECL(CXXDestructorDecl, {
    // We skip decls_begin/decls_end, which are already covered by
    // TraverseFunctionHelper().
    return TraverseFunctionHelper(D);
  })

template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseVarHelper(VarDecl *D) {
  TRY_TO(TraverseDeclaratorHelper(D));
  // Default params are taken care of when we traverse the ParmVarDecl.
  if (!isa<ParmVarDecl>(D) &&
      (!D->isCXXForRangeDecl() || getDerived().shouldVisitImplicitCode()))
    TRY_TO(TraverseStmt(D->getInit()));
  return true;
}

DEF_TRAVERSE_DECL(VarDecl, {
    TRY_TO(TraverseVarHelper(D));
  })

DEF_TRAVERSE_DECL(ImplicitParamDecl, {
    TRY_TO(TraverseVarHelper(D));
  })

DEF_TRAVERSE_DECL(NonTypeTemplateParmDecl, {
    // A non-type template parameter, e.g. "S" in template<int S> class Foo ...
    TRY_TO(TraverseDeclaratorHelper(D));
    if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
      TRY_TO(TraverseStmt(D->getDefaultArgument()));
  })

DEF_TRAVERSE_DECL(ParmVarDecl, {
    TRY_TO(TraverseVarHelper(D));

    if (D->hasDefaultArg() &&
        D->hasUninstantiatedDefaultArg() &&
        !D->hasUnparsedDefaultArg())
      TRY_TO(TraverseStmt(D->getUninstantiatedDefaultArg()));

    if (D->hasDefaultArg() &&
        !D->hasUninstantiatedDefaultArg() &&
        !D->hasUnparsedDefaultArg())
      TRY_TO(TraverseStmt(D->getDefaultArg()));
  })

#undef DEF_TRAVERSE_DECL

// ----------------- Stmt traversal -----------------
//
// For stmts, we automate (in the DEF_TRAVERSE_STMT macro) iterating
// over the children defined in children() (every stmt defines these,
// though sometimes the range is empty).  Each individual Traverse*
// method only needs to worry about children other than those.  To see
// what children() does for a given class, see, e.g.,
//   http://clang.llvm.org/doxygen/Stmt_8cpp_source.html

// This macro makes available a variable S, the passed-in stmt.
#define DEF_TRAVERSE_STMT(STMT, CODE)                                   \
template<typename Derived>                                              \
bool RecursiveASTVisitor<Derived>::Traverse##STMT (STMT *S) {           \
  TRY_TO(WalkUpFrom##STMT(S));                                          \
  { CODE; }                                                             \
  for (Stmt::child_range range = S->children(); range; ++range) {       \
    TRY_TO(TraverseStmt(*range));                                       \
  }                                                                     \
  return true;                                                          \
}

DEF_TRAVERSE_STMT(GCCAsmStmt, {
    TRY_TO(TraverseStmt(S->getAsmString()));
    for (unsigned I = 0, E = S->getNumInputs(); I < E; ++I) {
      TRY_TO(TraverseStmt(S->getInputConstraintLiteral(I)));
    }
    for (unsigned I = 0, E = S->getNumOutputs(); I < E; ++I) {
      TRY_TO(TraverseStmt(S->getOutputConstraintLiteral(I)));
    }
    for (unsigned I = 0, E = S->getNumClobbers(); I < E; ++I) {
      TRY_TO(TraverseStmt(S->getClobberStringLiteral(I)));
    }
    // children() iterates over inputExpr and outputExpr.
  })

DEF_TRAVERSE_STMT(MSAsmStmt, { 
    // FIXME: MS Asm doesn't currently parse Constraints, Clobbers, etc.  Once
    // added this needs to be implemented.
  })

DEF_TRAVERSE_STMT(CXXCatchStmt, {
    TRY_TO(TraverseDecl(S->getExceptionDecl()));
    // children() iterates over the handler block.
  })

DEF_TRAVERSE_STMT(DeclStmt, {
    for (DeclStmt::decl_iterator I = S->decl_begin(), E = S->decl_end();
         I != E; ++I) {
      TRY_TO(TraverseDecl(*I));
    }
    // Suppress the default iteration over children() by
    // returning.  Here's why: A DeclStmt looks like 'type var [=
    // initializer]'.  The decls above already traverse over the
    // initializers, so we don't have to do it again (which
    // children() would do).
    return true;
  })


// These non-expr stmts (most of them), do not need any action except
// iterating over the children.
DEF_TRAVERSE_STMT(BreakStmt, { })
DEF_TRAVERSE_STMT(CXXTryStmt, { })
DEF_TRAVERSE_STMT(CaseStmt, { })
DEF_TRAVERSE_STMT(CompoundStmt, { })
DEF_TRAVERSE_STMT(ContinueStmt, { })
DEF_TRAVERSE_STMT(DefaultStmt, { })
DEF_TRAVERSE_STMT(DoStmt, { })
DEF_TRAVERSE_STMT(ForStmt, { })
DEF_TRAVERSE_STMT(GotoStmt, { })
DEF_TRAVERSE_STMT(IfStmt, { })
DEF_TRAVERSE_STMT(IndirectGotoStmt, { })
DEF_TRAVERSE_STMT(LabelStmt, { })
DEF_TRAVERSE_STMT(AttributedStmt, { })
DEF_TRAVERSE_STMT(NullStmt, { })
DEF_TRAVERSE_STMT(ObjCAtCatchStmt, { })
DEF_TRAVERSE_STMT(ObjCAtFinallyStmt, { })
DEF_TRAVERSE_STMT(ObjCAtSynchronizedStmt, { })
DEF_TRAVERSE_STMT(ObjCAtThrowStmt, { })
DEF_TRAVERSE_STMT(ObjCAtTryStmt, { })
DEF_TRAVERSE_STMT(ObjCForCollectionStmt, { })
DEF_TRAVERSE_STMT(ObjCAutoreleasePoolStmt, { })
DEF_TRAVERSE_STMT(CXXForRangeStmt, {
  if (!getDerived().shouldVisitImplicitCode()) {
    TRY_TO(TraverseStmt(S->getLoopVarStmt()));
    TRY_TO(TraverseStmt(S->getRangeInit()));
    TRY_TO(TraverseStmt(S->getBody()));
    // Visit everything else only if shouldVisitImplicitCode().
    return true;
  }
})
DEF_TRAVERSE_STMT(MSDependentExistsStmt, {
    TRY_TO(TraverseNestedNameSpecifierLoc(S->getQualifierLoc()));
    TRY_TO(TraverseDeclarationNameInfo(S->getNameInfo()));
})
DEF_TRAVERSE_STMT(ReturnStmt, { })
DEF_TRAVERSE_STMT(SwitchStmt, { })
DEF_TRAVERSE_STMT(WhileStmt, { })


DEF_TRAVERSE_STMT(CXXDependentScopeMemberExpr, {
    TRY_TO(TraverseNestedNameSpecifierLoc(S->getQualifierLoc()));
    TRY_TO(TraverseDeclarationNameInfo(S->getMemberNameInfo()));
    if (S->hasExplicitTemplateArgs()) {
      TRY_TO(TraverseTemplateArgumentLocsHelper(
          S->getTemplateArgs(), S->getNumTemplateArgs()));
    }
  })

DEF_TRAVERSE_STMT(DeclRefExpr, {
    TRY_TO(TraverseNestedNameSpecifierLoc(S->getQualifierLoc()));
    TRY_TO(TraverseDeclarationNameInfo(S->getNameInfo()));
    TRY_TO(TraverseTemplateArgumentLocsHelper(
        S->getTemplateArgs(), S->getNumTemplateArgs()));
  })

DEF_TRAVERSE_STMT(DependentScopeDeclRefExpr, {
    TRY_TO(TraverseNestedNameSpecifierLoc(S->getQualifierLoc()));
    TRY_TO(TraverseDeclarationNameInfo(S->getNameInfo()));
    if (S->hasExplicitTemplateArgs()) {
      TRY_TO(TraverseTemplateArgumentLocsHelper(
          S->getExplicitTemplateArgs().getTemplateArgs(),
          S->getNumTemplateArgs()));
    }
  })

DEF_TRAVERSE_STMT(MemberExpr, {
    TRY_TO(TraverseNestedNameSpecifierLoc(S->getQualifierLoc()));
    TRY_TO(TraverseDeclarationNameInfo(S->getMemberNameInfo()));
    TRY_TO(TraverseTemplateArgumentLocsHelper(
        S->getTemplateArgs(), S->getNumTemplateArgs()));
  })

DEF_TRAVERSE_STMT(ImplicitCastExpr, {
    // We don't traverse the cast type, as it's not written in the
    // source code.
  })

DEF_TRAVERSE_STMT(CStyleCastExpr, {
    TRY_TO(TraverseTypeLoc(S->getTypeInfoAsWritten()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(CXXFunctionalCastExpr, {
    TRY_TO(TraverseTypeLoc(S->getTypeInfoAsWritten()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(CXXConstCastExpr, {
    TRY_TO(TraverseTypeLoc(S->getTypeInfoAsWritten()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(CXXDynamicCastExpr, {
    TRY_TO(TraverseTypeLoc(S->getTypeInfoAsWritten()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(CXXReinterpretCastExpr, {
    TRY_TO(TraverseTypeLoc(S->getTypeInfoAsWritten()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(CXXStaticCastExpr, {
    TRY_TO(TraverseTypeLoc(S->getTypeInfoAsWritten()->getTypeLoc()));
  })

// InitListExpr is a tricky one, because we want to do all our work on
// the syntactic form of the listexpr, but this method takes the
// semantic form by default.  We can't use the macro helper because it
// calls WalkUp*() on the semantic form, before our code can convert
// to the syntactic form.
template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseInitListExpr(InitListExpr *S) {
  if (InitListExpr *Syn = S->getSyntacticForm())
    S = Syn;
  TRY_TO(WalkUpFromInitListExpr(S));
  // All we need are the default actions.  FIXME: use a helper function.
  for (Stmt::child_range range = S->children(); range; ++range) {
    TRY_TO(TraverseStmt(*range));
  }
  return true;
}

// GenericSelectionExpr is a special case because the types and expressions
// are interleaved.  We also need to watch out for null types (default
// generic associations).
template<typename Derived>
bool RecursiveASTVisitor<Derived>::
TraverseGenericSelectionExpr(GenericSelectionExpr *S) {
  TRY_TO(WalkUpFromGenericSelectionExpr(S));
  TRY_TO(TraverseStmt(S->getControllingExpr()));
  for (unsigned i = 0; i != S->getNumAssocs(); ++i) {
    if (TypeSourceInfo *TS = S->getAssocTypeSourceInfo(i))
      TRY_TO(TraverseTypeLoc(TS->getTypeLoc()));
    TRY_TO(TraverseStmt(S->getAssocExpr(i)));
  }
  return true;
}

// PseudoObjectExpr is a special case because of the wierdness with
// syntactic expressions and opaque values.
template<typename Derived>
bool RecursiveASTVisitor<Derived>::
TraversePseudoObjectExpr(PseudoObjectExpr *S) {
  TRY_TO(WalkUpFromPseudoObjectExpr(S));
  TRY_TO(TraverseStmt(S->getSyntacticForm()));
  for (PseudoObjectExpr::semantics_iterator
         i = S->semantics_begin(), e = S->semantics_end(); i != e; ++i) {
    Expr *sub = *i;
    if (OpaqueValueExpr *OVE = dyn_cast<OpaqueValueExpr>(sub))
      sub = OVE->getSourceExpr();
    TRY_TO(TraverseStmt(sub));
  }
  return true;
}

DEF_TRAVERSE_STMT(CXXScalarValueInitExpr, {
    // This is called for code like 'return T()' where T is a built-in
    // (i.e. non-class) type.
    TRY_TO(TraverseTypeLoc(S->getTypeSourceInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(CXXNewExpr, {
  // The child-iterator will pick up the other arguments.
  TRY_TO(TraverseTypeLoc(S->getAllocatedTypeSourceInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(OffsetOfExpr, {
    // The child-iterator will pick up the expression representing
    // the field.
    // FIMXE: for code like offsetof(Foo, a.b.c), should we get
    // making a MemberExpr callbacks for Foo.a, Foo.a.b, and Foo.a.b.c?
    TRY_TO(TraverseTypeLoc(S->getTypeSourceInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(UnaryExprOrTypeTraitExpr, {
    // The child-iterator will pick up the arg if it's an expression,
    // but not if it's a type.
    if (S->isArgumentType())
      TRY_TO(TraverseTypeLoc(S->getArgumentTypeInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(CXXTypeidExpr, {
    // The child-iterator will pick up the arg if it's an expression,
    // but not if it's a type.
    if (S->isTypeOperand())
      TRY_TO(TraverseTypeLoc(S->getTypeOperandSourceInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(CXXUuidofExpr, {
    // The child-iterator will pick up the arg if it's an expression,
    // but not if it's a type.
    if (S->isTypeOperand())
      TRY_TO(TraverseTypeLoc(S->getTypeOperandSourceInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(UnaryTypeTraitExpr, {
    TRY_TO(TraverseTypeLoc(S->getQueriedTypeSourceInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(BinaryTypeTraitExpr, {
    TRY_TO(TraverseTypeLoc(S->getLhsTypeSourceInfo()->getTypeLoc()));
    TRY_TO(TraverseTypeLoc(S->getRhsTypeSourceInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(TypeTraitExpr, {
  for (unsigned I = 0, N = S->getNumArgs(); I != N; ++I)
    TRY_TO(TraverseTypeLoc(S->getArg(I)->getTypeLoc()));
})

DEF_TRAVERSE_STMT(ArrayTypeTraitExpr, {
    TRY_TO(TraverseTypeLoc(S->getQueriedTypeSourceInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(ExpressionTraitExpr, {
    TRY_TO(TraverseStmt(S->getQueriedExpression()));
  })

DEF_TRAVERSE_STMT(VAArgExpr, {
    // The child-iterator will pick up the expression argument.
    TRY_TO(TraverseTypeLoc(S->getWrittenTypeInfo()->getTypeLoc()));
  })

DEF_TRAVERSE_STMT(CXXTemporaryObjectExpr, {
    // This is called for code like 'return T()' where T is a class type.
    TRY_TO(TraverseTypeLoc(S->getTypeSourceInfo()->getTypeLoc()));
  })

// Walk only the visible parts of lambda expressions.  
template<typename Derived>
bool RecursiveASTVisitor<Derived>::TraverseLambdaExpr(LambdaExpr *S) {
  for (LambdaExpr::capture_iterator C = S->explicit_capture_begin(),
                                 CEnd = S->explicit_capture_end();
       C != CEnd; ++C) {
    TRY_TO(TraverseLambdaCapture(*C));
  }

  if (S->hasExplicitParameters() || S->hasExplicitResultType()) {
    TypeLoc TL = S->getCallOperator()->getTypeSourceInfo()->getTypeLoc();
    if (S->hasExplicitParameters() && S->hasExplicitResultType()) {
      // Visit the whole type.
      TRY_TO(TraverseTypeLoc(TL));
    } else if (isa<FunctionProtoTypeLoc>(TL)) {
      FunctionProtoTypeLoc Proto = cast<FunctionProtoTypeLoc>(TL);
      if (S->hasExplicitParameters()) {
        // Visit parameters.
        for (unsigned I = 0, N = Proto.getNumArgs(); I != N; ++I) {
          TRY_TO(TraverseDecl(Proto.getArg(I)));
        }
      } else {
        TRY_TO(TraverseTypeLoc(Proto.getResultLoc()));
      }        
    }
  }

  TRY_TO(TraverseStmt(S->getBody()));
  return true;
}

DEF_TRAVERSE_STMT(CXXUnresolvedConstructExpr, {
    // This is called for code like 'T()', where T is a template argument.
    TRY_TO(TraverseTypeLoc(S->getTypeSourceInfo()->getTypeLoc()));
  })

// These expressions all might take explicit template arguments.
// We traverse those if so.  FIXME: implement these.
DEF_TRAVERSE_STMT(CXXConstructExpr, { })
DEF_TRAVERSE_STMT(CallExpr, { })
DEF_TRAVERSE_STMT(CXXMemberCallExpr, { })

// These exprs (most of them), do not need any action except iterating
// over the children.
DEF_TRAVERSE_STMT(AddrLabelExpr, { })
DEF_TRAVERSE_STMT(ArraySubscriptExpr, { })
DEF_TRAVERSE_STMT(BlockExpr, {
  TRY_TO(TraverseDecl(S->getBlockDecl()));
  return true; // no child statements to loop through.
})
DEF_TRAVERSE_STMT(ChooseExpr, { })
DEF_TRAVERSE_STMT(CompoundLiteralExpr, {
  TRY_TO(TraverseTypeLoc(S->getTypeSourceInfo()->getTypeLoc()));
})
DEF_TRAVERSE_STMT(CXXBindTemporaryExpr, { })
DEF_TRAVERSE_STMT(CXXBoolLiteralExpr, { })
DEF_TRAVERSE_STMT(CXXDefaultArgExpr, { })
DEF_TRAVERSE_STMT(CXXDeleteExpr, { })
DEF_TRAVERSE_STMT(ExprWithCleanups, { })
DEF_TRAVERSE_STMT(CXXNullPtrLiteralExpr, { })
DEF_TRAVERSE_STMT(CXXPseudoDestructorExpr, {
  TRY_TO(TraverseNestedNameSpecifierLoc(S->getQualifierLoc()));
  if (TypeSourceInfo *ScopeInfo = S->getScopeTypeInfo())
    TRY_TO(TraverseTypeLoc(ScopeInfo->getTypeLoc()));
  if (TypeSourceInfo *DestroyedTypeInfo = S->getDestroyedTypeInfo())
    TRY_TO(TraverseTypeLoc(DestroyedTypeInfo->getTypeLoc()));
})
DEF_TRAVERSE_STMT(CXXThisExpr, { })
DEF_TRAVERSE_STMT(CXXThrowExpr, { })
DEF_TRAVERSE_STMT(UserDefinedLiteral, { })
DEF_TRAVERSE_STMT(DesignatedInitExpr, { })
DEF_TRAVERSE_STMT(ExtVectorElementExpr, { })
DEF_TRAVERSE_STMT(GNUNullExpr, { })
DEF_TRAVERSE_STMT(ImplicitValueInitExpr, { })
DEF_TRAVERSE_STMT(ObjCBoolLiteralExpr, { })
DEF_TRAVERSE_STMT(ObjCEncodeExpr, {
  if (TypeSourceInfo *TInfo = S->getEncodedTypeSourceInfo())
    TRY_TO(TraverseTypeLoc(TInfo->getTypeLoc()));
})
DEF_TRAVERSE_STMT(ObjCIsaExpr, { })
DEF_TRAVERSE_STMT(ObjCIvarRefExpr, { })
DEF_TRAVERSE_STMT(ObjCMessageExpr, { })
DEF_TRAVERSE_STMT(ObjCPropertyRefExpr, { })
DEF_TRAVERSE_STMT(ObjCSubscriptRefExpr, { })
DEF_TRAVERSE_STMT(ObjCProtocolExpr, { })
DEF_TRAVERSE_STMT(ObjCSelectorExpr, { })
DEF_TRAVERSE_STMT(ObjCIndirectCopyRestoreExpr, { })
DEF_TRAVERSE_STMT(ObjCBridgedCastExpr, {
  TRY_TO(TraverseTypeLoc(S->getTypeInfoAsWritten()->getTypeLoc()));
})
DEF_TRAVERSE_STMT(ParenExpr, { })
DEF_TRAVERSE_STMT(ParenListExpr, { })
DEF_TRAVERSE_STMT(PredefinedExpr, { })
DEF_TRAVERSE_STMT(ShuffleVectorExpr, { })
DEF_TRAVERSE_STMT(StmtExpr, { })
DEF_TRAVERSE_STMT(UnresolvedLookupExpr, {
  TRY_TO(TraverseNestedNameSpecifierLoc(S->getQualifierLoc()));
  if (S->hasExplicitTemplateArgs()) {
    TRY_TO(TraverseTemplateArgumentLocsHelper(S->getTemplateArgs(),
                                              S->getNumTemplateArgs()));
  }
})

DEF_TRAVERSE_STMT(UnresolvedMemberExpr, {
  TRY_TO(TraverseNestedNameSpecifierLoc(S->getQualifierLoc()));
  if (S->hasExplicitTemplateArgs()) {
    TRY_TO(TraverseTemplateArgumentLocsHelper(S->getTemplateArgs(),
                                              S->getNumTemplateArgs()));
  }
})

DEF_TRAVERSE_STMT(SEHTryStmt, {})
DEF_TRAVERSE_STMT(SEHExceptStmt, {})
DEF_TRAVERSE_STMT(SEHFinallyStmt,{})

DEF_TRAVERSE_STMT(CXXOperatorCallExpr, { })
DEF_TRAVERSE_STMT(OpaqueValueExpr, { })
DEF_TRAVERSE_STMT(CUDAKernelCallExpr, { })

// These operators (all of them) do not need any action except
// iterating over the children.
DEF_TRAVERSE_STMT(BinaryConditionalOperator, { })
DEF_TRAVERSE_STMT(ConditionalOperator, { })
DEF_TRAVERSE_STMT(UnaryOperator, { })
DEF_TRAVERSE_STMT(BinaryOperator, { })
DEF_TRAVERSE_STMT(CompoundAssignOperator, { })
DEF_TRAVERSE_STMT(CXXNoexceptExpr, { })
DEF_TRAVERSE_STMT(PackExpansionExpr, { })
DEF_TRAVERSE_STMT(SizeOfPackExpr, { })
DEF_TRAVERSE_STMT(SubstNonTypeTemplateParmPackExpr, { })
DEF_TRAVERSE_STMT(SubstNonTypeTemplateParmExpr, { })
DEF_TRAVERSE_STMT(FunctionParmPackExpr, { })
DEF_TRAVERSE_STMT(MaterializeTemporaryExpr, { })
DEF_TRAVERSE_STMT(AtomicExpr, { })

// These literals (all of them) do not need any action.
DEF_TRAVERSE_STMT(IntegerLiteral, { })
DEF_TRAVERSE_STMT(CharacterLiteral, { })
DEF_TRAVERSE_STMT(FloatingLiteral, { })
DEF_TRAVERSE_STMT(ImaginaryLiteral, { })
DEF_TRAVERSE_STMT(StringLiteral, { })
DEF_TRAVERSE_STMT(ObjCStringLiteral, { })
DEF_TRAVERSE_STMT(ObjCBoxedExpr, { })
DEF_TRAVERSE_STMT(ObjCArrayLiteral, { })
DEF_TRAVERSE_STMT(ObjCDictionaryLiteral, { })
  
// Traverse OpenCL: AsType, Convert.
DEF_TRAVERSE_STMT(AsTypeExpr, { })

// FIXME: look at the following tricky-seeming exprs to see if we
// need to recurse on anything.  These are ones that have methods
// returning decls or qualtypes or nestednamespecifier -- though I'm
// not sure if they own them -- or just seemed very complicated, or
// had lots of sub-types to explore.
//
// VisitOverloadExpr and its children: recurse on template args? etc?

// FIXME: go through all the stmts and exprs again, and see which of them
// create new types, and recurse on the types (TypeLocs?) of those.
// Candidates:
//
//    http://clang.llvm.org/doxygen/classclang_1_1CXXTypeidExpr.html
//    http://clang.llvm.org/doxygen/classclang_1_1UnaryExprOrTypeTraitExpr.html
//    http://clang.llvm.org/doxygen/classclang_1_1TypesCompatibleExpr.html
//    Every class that has getQualifier.

#undef DEF_TRAVERSE_STMT

#undef TRY_TO

} // end namespace clang

#endif // LLVM_CLANG_AST_RECURSIVEASTVISITOR_H
