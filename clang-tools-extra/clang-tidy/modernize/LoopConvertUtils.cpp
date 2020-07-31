//===--- LoopConvertUtils.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoopConvertUtils.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Lambda.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <utility>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

/// Tracks a stack of parent statements during traversal.
///
/// All this really does is inject push_back() before running
/// RecursiveASTVisitor::TraverseStmt() and pop_back() afterwards. The Stmt atop
/// the stack is the parent of the current statement (NULL for the topmost
/// statement).
bool StmtAncestorASTVisitor::TraverseStmt(Stmt *Statement) {
  StmtAncestors.insert(std::make_pair(Statement, StmtStack.back()));
  StmtStack.push_back(Statement);
  RecursiveASTVisitor<StmtAncestorASTVisitor>::TraverseStmt(Statement);
  StmtStack.pop_back();
  return true;
}

/// Keep track of the DeclStmt associated with each VarDecl.
///
/// Combined with StmtAncestors, this provides roughly the same information as
/// Scope, as we can map a VarDecl to its DeclStmt, then walk up the parent tree
/// using StmtAncestors.
bool StmtAncestorASTVisitor::VisitDeclStmt(DeclStmt *Decls) {
  for (const auto *decl : Decls->decls()) {
    if (const auto *V = dyn_cast<VarDecl>(decl))
      DeclParents.insert(std::make_pair(V, Decls));
  }
  return true;
}

/// record the DeclRefExpr as part of the parent expression.
bool ComponentFinderASTVisitor::VisitDeclRefExpr(DeclRefExpr *E) {
  Components.push_back(E);
  return true;
}

/// record the MemberExpr as part of the parent expression.
bool ComponentFinderASTVisitor::VisitMemberExpr(MemberExpr *Member) {
  Components.push_back(Member);
  return true;
}

/// Forward any DeclRefExprs to a check on the referenced variable
/// declaration.
bool DependencyFinderASTVisitor::VisitDeclRefExpr(DeclRefExpr *DeclRef) {
  if (auto *V = dyn_cast_or_null<VarDecl>(DeclRef->getDecl()))
    return VisitVarDecl(V);
  return true;
}

/// Determine if any this variable is declared inside the ContainingStmt.
bool DependencyFinderASTVisitor::VisitVarDecl(VarDecl *V) {
  const Stmt *Curr = DeclParents->lookup(V);
  // First, see if the variable was declared within an inner scope of the loop.
  while (Curr != nullptr) {
    if (Curr == ContainingStmt) {
      DependsOnInsideVariable = true;
      return false;
    }
    Curr = StmtParents->lookup(Curr);
  }

  // Next, check if the variable was removed from existence by an earlier
  // iteration.
  for (const auto &I : *ReplacedVars) {
    if (I.second == V) {
      DependsOnInsideVariable = true;
      return false;
    }
  }
  return true;
}

/// If we already created a variable for TheLoop, check to make sure
/// that the name was not already taken.
bool DeclFinderASTVisitor::VisitForStmt(ForStmt *TheLoop) {
  StmtGeneratedVarNameMap::const_iterator I = GeneratedDecls->find(TheLoop);
  if (I != GeneratedDecls->end() && I->second == Name) {
    Found = true;
    return false;
  }
  return true;
}

/// If any named declaration within the AST subtree has the same name,
/// then consider Name already taken.
bool DeclFinderASTVisitor::VisitNamedDecl(NamedDecl *D) {
  const IdentifierInfo *Ident = D->getIdentifier();
  if (Ident && Ident->getName() == Name) {
    Found = true;
    return false;
  }
  return true;
}

/// Forward any declaration references to the actual check on the
/// referenced declaration.
bool DeclFinderASTVisitor::VisitDeclRefExpr(DeclRefExpr *DeclRef) {
  if (auto *D = dyn_cast<NamedDecl>(DeclRef->getDecl()))
    return VisitNamedDecl(D);
  return true;
}

/// If the new variable name conflicts with any type used in the loop,
/// then we mark that variable name as taken.
bool DeclFinderASTVisitor::VisitTypeLoc(TypeLoc TL) {
  QualType QType = TL.getType();

  // Check if our name conflicts with a type, to handle for typedefs.
  if (QType.getAsString() == Name) {
    Found = true;
    return false;
  }
  // Check for base type conflicts. For example, when a struct is being
  // referenced in the body of the loop, the above getAsString() will return the
  // whole type (ex. "struct s"), but will be caught here.
  if (const IdentifierInfo *Ident = QType.getBaseTypeIdentifier()) {
    if (Ident->getName() == Name) {
      Found = true;
      return false;
    }
  }
  return true;
}

/// Look through conversion/copy constructors to find the explicit
/// initialization expression, returning it is found.
///
/// The main idea is that given
///   vector<int> v;
/// we consider either of these initializations
///   vector<int>::iterator it = v.begin();
///   vector<int>::iterator it(v.begin());
/// and retrieve `v.begin()` as the expression used to initialize `it` but do
/// not include
///   vector<int>::iterator it;
///   vector<int>::iterator it(v.begin(), 0); // if this constructor existed
/// as being initialized from `v.begin()`
const Expr *digThroughConstructors(const Expr *E) {
  if (!E)
    return nullptr;
  E = E->IgnoreImplicit();
  if (const auto *ConstructExpr = dyn_cast<CXXConstructExpr>(E)) {
    // The initial constructor must take exactly one parameter, but base class
    // and deferred constructors can take more.
    if (ConstructExpr->getNumArgs() != 1 ||
        ConstructExpr->getConstructionKind() != CXXConstructExpr::CK_Complete)
      return nullptr;
    E = ConstructExpr->getArg(0);
    if (const auto *Temp = dyn_cast<MaterializeTemporaryExpr>(E))
      E = Temp->getSubExpr();
    return digThroughConstructors(E);
  }
  return E;
}

/// Returns true when two Exprs are equivalent.
bool areSameExpr(ASTContext *Context, const Expr *First, const Expr *Second) {
  if (!First || !Second)
    return false;

  llvm::FoldingSetNodeID FirstID, SecondID;
  First->Profile(FirstID, *Context, true);
  Second->Profile(SecondID, *Context, true);
  return FirstID == SecondID;
}

/// Returns the DeclRefExpr represented by E, or NULL if there isn't one.
const DeclRefExpr *getDeclRef(const Expr *E) {
  return dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts());
}

/// Returns true when two ValueDecls are the same variable.
bool areSameVariable(const ValueDecl *First, const ValueDecl *Second) {
  return First && Second &&
         First->getCanonicalDecl() == Second->getCanonicalDecl();
}

/// Determines if an expression is a declaration reference to a
/// particular variable.
static bool exprReferencesVariable(const ValueDecl *Target, const Expr *E) {
  if (!Target || !E)
    return false;
  const DeclRefExpr *Decl = getDeclRef(E);
  return Decl && areSameVariable(Target, Decl->getDecl());
}

/// If the expression is a dereference or call to operator*(), return the
/// operand. Otherwise, return NULL.
static const Expr *getDereferenceOperand(const Expr *E) {
  if (const auto *Uop = dyn_cast<UnaryOperator>(E))
    return Uop->getOpcode() == UO_Deref ? Uop->getSubExpr() : nullptr;

  if (const auto *OpCall = dyn_cast<CXXOperatorCallExpr>(E)) {
    return OpCall->getOperator() == OO_Star && OpCall->getNumArgs() == 1
               ? OpCall->getArg(0)
               : nullptr;
  }

  return nullptr;
}

/// Returns true when the Container contains an Expr equivalent to E.
template <typename ContainerT>
static bool containsExpr(ASTContext *Context, const ContainerT *Container,
                         const Expr *E) {
  llvm::FoldingSetNodeID ID;
  E->Profile(ID, *Context, true);
  for (const auto &I : *Container) {
    if (ID == I.second)
      return true;
  }
  return false;
}

/// Returns true when the index expression is a declaration reference to
/// IndexVar.
///
/// If the index variable is `index`, this function returns true on
///    arrayExpression[index];
///    containerExpression[index];
/// but not
///    containerExpression[notIndex];
static bool isIndexInSubscriptExpr(const Expr *IndexExpr,
                                   const VarDecl *IndexVar) {
  const DeclRefExpr *Idx = getDeclRef(IndexExpr);
  return Idx && Idx->getType()->isIntegerType() &&
         areSameVariable(IndexVar, Idx->getDecl());
}

/// Returns true when the index expression is a declaration reference to
/// IndexVar, Obj is the same expression as SourceExpr after all parens and
/// implicit casts are stripped off.
///
/// If PermitDeref is true, IndexExpression may
/// be a dereference (overloaded or builtin operator*).
///
/// This function is intended for array-like containers, as it makes sure that
/// both the container and the index match.
/// If the loop has index variable `index` and iterates over `container`, then
/// isIndexInSubscriptExpr returns true for
/// \code
///   container[index]
///   container.at(index)
///   container->at(index)
/// \endcode
/// but not for
/// \code
///   container[notIndex]
///   notContainer[index]
/// \endcode
/// If PermitDeref is true, then isIndexInSubscriptExpr additionally returns
/// true on these expressions:
/// \code
///   (*container)[index]
///   (*container).at(index)
/// \endcode
static bool isIndexInSubscriptExpr(ASTContext *Context, const Expr *IndexExpr,
                                   const VarDecl *IndexVar, const Expr *Obj,
                                   const Expr *SourceExpr, bool PermitDeref) {
  if (!SourceExpr || !Obj || !isIndexInSubscriptExpr(IndexExpr, IndexVar))
    return false;

  if (areSameExpr(Context, SourceExpr->IgnoreParenImpCasts(),
                  Obj->IgnoreParenImpCasts()))
    return true;

  if (const Expr *InnerObj = getDereferenceOperand(Obj->IgnoreParenImpCasts()))
    if (PermitDeref && areSameExpr(Context, SourceExpr->IgnoreParenImpCasts(),
                                   InnerObj->IgnoreParenImpCasts()))
      return true;

  return false;
}

/// Returns true when Opcall is a call a one-parameter dereference of
/// IndexVar.
///
/// For example, if the index variable is `index`, returns true for
///   *index
/// but not
///   index
///   *notIndex
static bool isDereferenceOfOpCall(const CXXOperatorCallExpr *OpCall,
                                  const VarDecl *IndexVar) {
  return OpCall->getOperator() == OO_Star && OpCall->getNumArgs() == 1 &&
         exprReferencesVariable(IndexVar, OpCall->getArg(0));
}

/// Returns true when Uop is a dereference of IndexVar.
///
/// For example, if the index variable is `index`, returns true for
///   *index
/// but not
///   index
///   *notIndex
static bool isDereferenceOfUop(const UnaryOperator *Uop,
                               const VarDecl *IndexVar) {
  return Uop->getOpcode() == UO_Deref &&
         exprReferencesVariable(IndexVar, Uop->getSubExpr());
}

/// Determines whether the given Decl defines a variable initialized to
/// the loop object.
///
/// This is intended to find cases such as
/// \code
///   for (int i = 0; i < arraySize(arr); ++i) {
///     T t = arr[i];
///     // use t, do not use i
///   }
/// \endcode
/// and
/// \code
///   for (iterator i = container.begin(), e = container.end(); i != e; ++i) {
///     T t = *i;
///     // use t, do not use i
///   }
/// \endcode
static bool isAliasDecl(ASTContext *Context, const Decl *TheDecl,
                        const VarDecl *IndexVar) {
  const auto *VDecl = dyn_cast<VarDecl>(TheDecl);
  if (!VDecl)
    return false;
  if (!VDecl->hasInit())
    return false;

  bool OnlyCasts = true;
  const Expr *Init = VDecl->getInit()->IgnoreParenImpCasts();
  if (Init && isa<CXXConstructExpr>(Init)) {
    Init = digThroughConstructors(Init);
    OnlyCasts = false;
  }
  if (!Init)
    return false;

  // Check that the declared type is the same as (or a reference to) the
  // container type.
  if (!OnlyCasts) {
    QualType InitType = Init->getType();
    QualType DeclarationType = VDecl->getType();
    if (!DeclarationType.isNull() && DeclarationType->isReferenceType())
      DeclarationType = DeclarationType.getNonReferenceType();

    if (InitType.isNull() || DeclarationType.isNull() ||
        !Context->hasSameUnqualifiedType(DeclarationType, InitType))
      return false;
  }

  switch (Init->getStmtClass()) {
  case Stmt::ArraySubscriptExprClass: {
    const auto *E = cast<ArraySubscriptExpr>(Init);
    // We don't really care which array is used here. We check to make sure
    // it was the correct one later, since the AST will traverse it next.
    return isIndexInSubscriptExpr(E->getIdx(), IndexVar);
  }

  case Stmt::UnaryOperatorClass:
    return isDereferenceOfUop(cast<UnaryOperator>(Init), IndexVar);

  case Stmt::CXXOperatorCallExprClass: {
    const auto *OpCall = cast<CXXOperatorCallExpr>(Init);
    if (OpCall->getOperator() == OO_Star)
      return isDereferenceOfOpCall(OpCall, IndexVar);
    if (OpCall->getOperator() == OO_Subscript) {
      assert(OpCall->getNumArgs() == 2);
      return isIndexInSubscriptExpr(OpCall->getArg(1), IndexVar);
    }
    break;
  }

  case Stmt::CXXMemberCallExprClass: {
    const auto *MemCall = cast<CXXMemberCallExpr>(Init);
    // This check is needed because getMethodDecl can return nullptr if the
    // callee is a member function pointer.
    const auto *MDecl = MemCall->getMethodDecl();
    if (MDecl && !isa<CXXConversionDecl>(MDecl) &&
        MDecl->getNameAsString() == "at" && MemCall->getNumArgs() == 1) {
      return isIndexInSubscriptExpr(MemCall->getArg(0), IndexVar);
    }
    return false;
  }

  default:
    break;
  }
  return false;
}

/// Determines whether the bound of a for loop condition expression is
/// the same as the statically computable size of ArrayType.
///
/// Given
/// \code
///   const int N = 5;
///   int arr[N];
/// \endcode
/// This is intended to permit
/// \code
///   for (int i = 0; i < N; ++i) {  /* use arr[i] */ }
///   for (int i = 0; i < arraysize(arr); ++i) { /* use arr[i] */ }
/// \endcode
static bool arrayMatchesBoundExpr(ASTContext *Context,
                                  const QualType &ArrayType,
                                  const Expr *ConditionExpr) {
  if (!ConditionExpr || ConditionExpr->isValueDependent())
    return false;
  const ConstantArrayType *ConstType =
      Context->getAsConstantArrayType(ArrayType);
  if (!ConstType)
    return false;
  Optional<llvm::APSInt> ConditionSize =
      ConditionExpr->getIntegerConstantExpr(*Context);
  if (!ConditionSize)
    return false;
  llvm::APSInt ArraySize(ConstType->getSize());
  return llvm::APSInt::isSameValue(*ConditionSize, ArraySize);
}

ForLoopIndexUseVisitor::ForLoopIndexUseVisitor(ASTContext *Context,
                                               const VarDecl *IndexVar,
                                               const VarDecl *EndVar,
                                               const Expr *ContainerExpr,
                                               const Expr *ArrayBoundExpr,
                                               bool ContainerNeedsDereference)
    : Context(Context), IndexVar(IndexVar), EndVar(EndVar),
      ContainerExpr(ContainerExpr), ArrayBoundExpr(ArrayBoundExpr),
      ContainerNeedsDereference(ContainerNeedsDereference),
      OnlyUsedAsIndex(true), AliasDecl(nullptr),
      ConfidenceLevel(Confidence::CL_Safe), NextStmtParent(nullptr),
      CurrStmtParent(nullptr), ReplaceWithAliasUse(false),
      AliasFromForInit(false) {
  if (ContainerExpr)
    addComponent(ContainerExpr);
}

bool ForLoopIndexUseVisitor::findAndVerifyUsages(const Stmt *Body) {
  TraverseStmt(const_cast<Stmt *>(Body));
  return OnlyUsedAsIndex && ContainerExpr;
}

void ForLoopIndexUseVisitor::addComponents(const ComponentVector &Components) {
  // FIXME: add sort(on ID)+unique to avoid extra work.
  for (const auto &I : Components)
    addComponent(I);
}

void ForLoopIndexUseVisitor::addComponent(const Expr *E) {
  llvm::FoldingSetNodeID ID;
  const Expr *Node = E->IgnoreParenImpCasts();
  Node->Profile(ID, *Context, true);
  DependentExprs.push_back(std::make_pair(Node, ID));
}

void ForLoopIndexUseVisitor::addUsage(const Usage &U) {
  SourceLocation Begin = U.Range.getBegin();
  if (Begin.isMacroID())
    Begin = Context->getSourceManager().getSpellingLoc(Begin);

  if (UsageLocations.insert(Begin).second)
    Usages.push_back(U);
}

/// If the unary operator is a dereference of IndexVar, include it
/// as a valid usage and prune the traversal.
///
/// For example, if container.begin() and container.end() both return pointers
/// to int, this makes sure that the initialization for `k` is not counted as an
/// unconvertible use of the iterator `i`.
/// \code
///   for (int *i = container.begin(), *e = container.end(); i != e; ++i) {
///     int k = *i + 2;
///   }
/// \endcode
bool ForLoopIndexUseVisitor::TraverseUnaryOperator(UnaryOperator *Uop) {
  // If we dereference an iterator that's actually a pointer, count the
  // occurrence.
  if (isDereferenceOfUop(Uop, IndexVar)) {
    addUsage(Usage(Uop));
    return true;
  }

  return VisitorBase::TraverseUnaryOperator(Uop);
}

/// If the member expression is operator-> (overloaded or not) on
/// IndexVar, include it as a valid usage and prune the traversal.
///
/// For example, given
/// \code
///   struct Foo { int bar(); int x; };
///   vector<Foo> v;
/// \endcode
/// the following uses will be considered convertible:
/// \code
///   for (vector<Foo>::iterator i = v.begin(), e = v.end(); i != e; ++i) {
///     int b = i->bar();
///     int k = i->x + 1;
///   }
/// \endcode
/// though
/// \code
///   for (vector<Foo>::iterator i = v.begin(), e = v.end(); i != e; ++i) {
///     int k = i.insert(1);
///   }
///   for (vector<Foo>::iterator i = v.begin(), e = v.end(); i != e; ++i) {
///     int b = e->bar();
///   }
/// \endcode
/// will not.
bool ForLoopIndexUseVisitor::TraverseMemberExpr(MemberExpr *Member) {
  const Expr *Base = Member->getBase();
  const DeclRefExpr *Obj = getDeclRef(Base);
  const Expr *ResultExpr = Member;
  QualType ExprType;
  if (const auto *Call =
          dyn_cast<CXXOperatorCallExpr>(Base->IgnoreParenImpCasts())) {
    // If operator->() is a MemberExpr containing a CXXOperatorCallExpr, then
    // the MemberExpr does not have the expression we want. We therefore catch
    // that instance here.
    // For example, if vector<Foo>::iterator defines operator->(), then the
    // example `i->bar()` at the top of this function is a CXXMemberCallExpr
    // referring to `i->` as the member function called. We want just `i`, so
    // we take the argument to operator->() as the base object.
    if (Call->getOperator() == OO_Arrow) {
      assert(Call->getNumArgs() == 1 &&
             "Operator-> takes more than one argument");
      Obj = getDeclRef(Call->getArg(0));
      ResultExpr = Obj;
      ExprType = Call->getCallReturnType(*Context);
    }
  }

  if (Obj && exprReferencesVariable(IndexVar, Obj)) {
    // Member calls on the iterator with '.' are not allowed.
    if (!Member->isArrow()) {
      OnlyUsedAsIndex = false;
      return true;
    }

    if (ExprType.isNull())
      ExprType = Obj->getType();

    if (!ExprType->isPointerType())
      return false;

    // FIXME: This works around not having the location of the arrow operator.
    // Consider adding OperatorLoc to MemberExpr?
    SourceLocation ArrowLoc = Lexer::getLocForEndOfToken(
        Base->getExprLoc(), 0, Context->getSourceManager(),
        Context->getLangOpts());
    // If something complicated is happening (i.e. the next token isn't an
    // arrow), give up on making this work.
    if (ArrowLoc.isValid()) {
      addUsage(Usage(ResultExpr, Usage::UK_MemberThroughArrow,
                     SourceRange(Base->getExprLoc(), ArrowLoc)));
      return true;
    }
  }
  return VisitorBase::TraverseMemberExpr(Member);
}

/// If a member function call is the at() accessor on the container with
/// IndexVar as the single argument, include it as a valid usage and prune
/// the traversal.
///
/// Member calls on other objects will not be permitted.
/// Calls on the iterator object are not permitted, unless done through
/// operator->(). The one exception is allowing vector::at() for pseudoarrays.
bool ForLoopIndexUseVisitor::TraverseCXXMemberCallExpr(
    CXXMemberCallExpr *MemberCall) {
  auto *Member =
      dyn_cast<MemberExpr>(MemberCall->getCallee()->IgnoreParenImpCasts());
  if (!Member)
    return VisitorBase::TraverseCXXMemberCallExpr(MemberCall);

  // We specifically allow an accessor named "at" to let STL in, though
  // this is restricted to pseudo-arrays by requiring a single, integer
  // argument.
  const IdentifierInfo *Ident = Member->getMemberDecl()->getIdentifier();
  if (Ident && Ident->isStr("at") && MemberCall->getNumArgs() == 1) {
    if (isIndexInSubscriptExpr(Context, MemberCall->getArg(0), IndexVar,
                               Member->getBase(), ContainerExpr,
                               ContainerNeedsDereference)) {
      addUsage(Usage(MemberCall));
      return true;
    }
  }

  if (containsExpr(Context, &DependentExprs, Member->getBase()))
    ConfidenceLevel.lowerTo(Confidence::CL_Risky);

  return VisitorBase::TraverseCXXMemberCallExpr(MemberCall);
}

/// If an overloaded operator call is a dereference of IndexVar or
/// a subscript of the container with IndexVar as the single argument,
/// include it as a valid usage and prune the traversal.
///
/// For example, given
/// \code
///   struct Foo { int bar(); int x; };
///   vector<Foo> v;
///   void f(Foo);
/// \endcode
/// the following uses will be considered convertible:
/// \code
///   for (vector<Foo>::iterator i = v.begin(), e = v.end(); i != e; ++i) {
///     f(*i);
///   }
///   for (int i = 0; i < v.size(); ++i) {
///      int i = v[i] + 1;
///   }
/// \endcode
bool ForLoopIndexUseVisitor::TraverseCXXOperatorCallExpr(
    CXXOperatorCallExpr *OpCall) {
  switch (OpCall->getOperator()) {
  case OO_Star:
    if (isDereferenceOfOpCall(OpCall, IndexVar)) {
      addUsage(Usage(OpCall));
      return true;
    }
    break;

  case OO_Subscript:
    if (OpCall->getNumArgs() != 2)
      break;
    if (isIndexInSubscriptExpr(Context, OpCall->getArg(1), IndexVar,
                               OpCall->getArg(0), ContainerExpr,
                               ContainerNeedsDereference)) {
      addUsage(Usage(OpCall));
      return true;
    }
    break;

  default:
    break;
  }
  return VisitorBase::TraverseCXXOperatorCallExpr(OpCall);
}

/// If we encounter an array with IndexVar as the index of an
/// ArraySubscriptExpression, note it as a consistent usage and prune the
/// AST traversal.
///
/// For example, given
/// \code
///   const int N = 5;
///   int arr[N];
/// \endcode
/// This is intended to permit
/// \code
///   for (int i = 0; i < N; ++i) {  /* use arr[i] */ }
/// \endcode
/// but not
/// \code
///   for (int i = 0; i < N; ++i) {  /* use notArr[i] */ }
/// \endcode
/// and further checking needs to be done later to ensure that exactly one array
/// is referenced.
bool ForLoopIndexUseVisitor::TraverseArraySubscriptExpr(ArraySubscriptExpr *E) {
  Expr *Arr = E->getBase();
  if (!isIndexInSubscriptExpr(E->getIdx(), IndexVar))
    return VisitorBase::TraverseArraySubscriptExpr(E);

  if ((ContainerExpr &&
       !areSameExpr(Context, Arr->IgnoreParenImpCasts(),
                    ContainerExpr->IgnoreParenImpCasts())) ||
      !arrayMatchesBoundExpr(Context, Arr->IgnoreImpCasts()->getType(),
                             ArrayBoundExpr)) {
    // If we have already discovered the array being indexed and this isn't it
    // or this array doesn't match, mark this loop as unconvertible.
    OnlyUsedAsIndex = false;
    return VisitorBase::TraverseArraySubscriptExpr(E);
  }

  if (!ContainerExpr)
    ContainerExpr = Arr;

  addUsage(Usage(E));
  return true;
}

/// If we encounter a reference to IndexVar in an unpruned branch of the
/// traversal, mark this loop as unconvertible.
///
/// This determines the set of convertible loops: any usages of IndexVar
/// not explicitly considered convertible by this traversal will be caught by
/// this function.
///
/// Additionally, if the container expression is more complex than just a
/// DeclRefExpr, and some part of it is appears elsewhere in the loop, lower
/// our confidence in the transformation.
///
/// For example, these are not permitted:
/// \code
///   for (int i = 0; i < N; ++i) {  printf("arr[%d] = %d", i, arr[i]); }
///   for (vector<int>::iterator i = container.begin(), e = container.end();
///        i != e; ++i)
///     i.insert(0);
///   for (vector<int>::iterator i = container.begin(), e = container.end();
///        i != e; ++i)
///     if (i + 1 != e)
///       printf("%d", *i);
/// \endcode
///
/// And these will raise the risk level:
/// \code
///    int arr[10][20];
///    int l = 5;
///    for (int j = 0; j < 20; ++j)
///      int k = arr[l][j] + l; // using l outside arr[l] is considered risky
///    for (int i = 0; i < obj.getVector().size(); ++i)
///      obj.foo(10); // using `obj` is considered risky
/// \endcode
bool ForLoopIndexUseVisitor::VisitDeclRefExpr(DeclRefExpr *E) {
  const ValueDecl *TheDecl = E->getDecl();
  if (areSameVariable(IndexVar, TheDecl) ||
      exprReferencesVariable(IndexVar, E) || areSameVariable(EndVar, TheDecl) ||
      exprReferencesVariable(EndVar, E))
    OnlyUsedAsIndex = false;
  if (containsExpr(Context, &DependentExprs, E))
    ConfidenceLevel.lowerTo(Confidence::CL_Risky);
  return true;
}

/// If the loop index is captured by a lambda, replace this capture
/// by the range-for loop variable.
///
/// For example:
/// \code
///   for (int i = 0; i < N; ++i) {
///     auto f = [v, i](int k) {
///       printf("%d\n", v[i] + k);
///     };
///     f(v[i]);
///   }
/// \endcode
///
/// Will be replaced by:
/// \code
///   for (auto & elem : v) {
///     auto f = [v, elem](int k) {
///       printf("%d\n", elem + k);
///     };
///     f(elem);
///   }
/// \endcode
bool ForLoopIndexUseVisitor::TraverseLambdaCapture(LambdaExpr *LE,
                                                   const LambdaCapture *C,
                                                   Expr *Init) {
  if (C->capturesVariable()) {
    const VarDecl *VDecl = C->getCapturedVar();
    if (areSameVariable(IndexVar, cast<ValueDecl>(VDecl))) {
      // FIXME: if the index is captured, it will count as an usage and the
      // alias (if any) won't work, because it is only used in case of having
      // exactly one usage.
      addUsage(Usage(nullptr,
                     C->getCaptureKind() == LCK_ByCopy ? Usage::UK_CaptureByCopy
                                                       : Usage::UK_CaptureByRef,
                     C->getLocation()));
    }
  }
  return VisitorBase::TraverseLambdaCapture(LE, C, Init);
}

/// If we find that another variable is created just to refer to the loop
/// element, note it for reuse as the loop variable.
///
/// See the comments for isAliasDecl.
bool ForLoopIndexUseVisitor::VisitDeclStmt(DeclStmt *S) {
  if (!AliasDecl && S->isSingleDecl() &&
      isAliasDecl(Context, S->getSingleDecl(), IndexVar)) {
    AliasDecl = S;
    if (CurrStmtParent) {
      if (isa<IfStmt>(CurrStmtParent) || isa<WhileStmt>(CurrStmtParent) ||
          isa<SwitchStmt>(CurrStmtParent))
        ReplaceWithAliasUse = true;
      else if (isa<ForStmt>(CurrStmtParent)) {
        if (cast<ForStmt>(CurrStmtParent)->getConditionVariableDeclStmt() == S)
          ReplaceWithAliasUse = true;
        else
          // It's assumed S came the for loop's init clause.
          AliasFromForInit = true;
      }
    }
  }

  return true;
}

bool ForLoopIndexUseVisitor::TraverseStmt(Stmt *S) {
  // If this is an initialization expression for a lambda capture, prune the
  // traversal so that we don't end up diagnosing the contained DeclRefExpr as
  // inconsistent usage. No need to record the usage here -- this is done in
  // TraverseLambdaCapture().
  if (const auto *LE = dyn_cast_or_null<LambdaExpr>(NextStmtParent)) {
    // Any child of a LambdaExpr that isn't the body is an initialization
    // expression.
    if (S != LE->getBody()) {
      return true;
    }
  }

  // All this pointer swapping is a mechanism for tracking immediate parentage
  // of Stmts.
  const Stmt *OldNextParent = NextStmtParent;
  CurrStmtParent = NextStmtParent;
  NextStmtParent = S;
  bool Result = VisitorBase::TraverseStmt(S);
  NextStmtParent = OldNextParent;
  return Result;
}

std::string VariableNamer::createIndexName() {
  // FIXME: Add in naming conventions to handle:
  //  - How to handle conflicts.
  //  - An interactive process for naming.
  std::string IteratorName;
  StringRef ContainerName;
  if (TheContainer)
    ContainerName = TheContainer->getName();

  size_t Len = ContainerName.size();
  if (Len > 1 && ContainerName.endswith(Style == NS_UpperCase ? "S" : "s")) {
    IteratorName = std::string(ContainerName.substr(0, Len - 1));
    // E.g.: (auto thing : things)
    if (!declarationExists(IteratorName) || IteratorName == OldIndex->getName())
      return IteratorName;
  }

  if (Len > 2 && ContainerName.endswith(Style == NS_UpperCase ? "S_" : "s_")) {
    IteratorName = std::string(ContainerName.substr(0, Len - 2));
    // E.g.: (auto thing : things_)
    if (!declarationExists(IteratorName) || IteratorName == OldIndex->getName())
      return IteratorName;
  }

  return std::string(OldIndex->getName());
}

/// Determines whether or not the name \a Symbol conflicts with
/// language keywords or defined macros. Also checks if the name exists in
/// LoopContext, any of its parent contexts, or any of its child statements.
///
/// We also check to see if the same identifier was generated by this loop
/// converter in a loop nested within SourceStmt.
bool VariableNamer::declarationExists(StringRef Symbol) {
  assert(Context != nullptr && "Expected an ASTContext");
  IdentifierInfo &Ident = Context->Idents.get(Symbol);

  // Check if the symbol is not an identifier (ie. is a keyword or alias).
  if (!isAnyIdentifier(Ident.getTokenID()))
    return true;

  // Check for conflicting macro definitions.
  if (Ident.hasMacroDefinition())
    return true;

  // Determine if the symbol was generated in a parent context.
  for (const Stmt *S = SourceStmt; S != nullptr; S = ReverseAST->lookup(S)) {
    StmtGeneratedVarNameMap::const_iterator I = GeneratedDecls->find(S);
    if (I != GeneratedDecls->end() && I->second == Symbol)
      return true;
  }

  // FIXME: Rather than detecting conflicts at their usages, we should check the
  // parent context.
  // For some reason, lookup() always returns the pair (NULL, NULL) because its
  // StoredDeclsMap is not initialized (i.e. LookupPtr.getInt() is false inside
  // of DeclContext::lookup()). Why is this?

  // Finally, determine if the symbol was used in the loop or a child context.
  DeclFinderASTVisitor DeclFinder(std::string(Symbol), GeneratedDecls);
  return DeclFinder.findUsages(SourceStmt);
}

} // namespace modernize
} // namespace tidy
} // namespace clang
