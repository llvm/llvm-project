//===-- LoopConvert/LoopActions.cpp - C++11 For loop migration ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines matchers and callbacks for use in migrating C++
/// for loops.
///
//===----------------------------------------------------------------------===//

#include "LoopActions.h"
#include "LoopMatchers.h"
#include "VariableNaming.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

/// \brief The information needed to describe a valid convertible usage
/// of an array index or iterator.
struct Usage {
  const Expr *E;
  bool IsArrow;
  SourceRange Range;

  explicit Usage(const Expr *E)
      : E(E), IsArrow(false), Range(E->getSourceRange()) { }
  Usage(const Expr *E, bool IsArrow, SourceRange Range)
      : E(E), IsArrow(IsArrow), Range(Range) { }
};

/// \brief A class to encapsulate lowering of the tool's confidence level.
///
/// Confidence is a quantity opposite in meaning to Risk. Since clang-modernize
/// uses risk, this class reverses the meaning for the legacy loop convert
/// code.
class Confidence {
public:
  /// \brief Initialize confidence level.
  explicit Confidence(RiskLevel Level) :
    CurrentLevel(Level) {}

  /// \brief Lower the internal confidence level to Level, but do not raise it.
  void lowerTo(RiskLevel Level) {
    CurrentLevel = std::max(Level, CurrentLevel);
  }

  /// \brief Return the internal confidence level.
  RiskLevel getRiskLevel() const { return CurrentLevel; }

private:
  RiskLevel CurrentLevel;
};

/// \brief Discover usages of expressions consisting of index or iterator
/// access.
///
/// Given an index variable, recursively crawls a for loop to discover if the
/// index variable is used in a way consistent with range-based for loop access.
class ForLoopIndexUseVisitor
    : public RecursiveASTVisitor<ForLoopIndexUseVisitor> {
 public:
  ForLoopIndexUseVisitor(ASTContext *Context, const VarDecl *IndexVar,
                         const VarDecl *EndVar, const Expr *ContainerExpr,
                         const Expr *ArrayBoundExpr,
                         bool ContainerNeedsDereference) :
    Context(Context), IndexVar(IndexVar), EndVar(EndVar),
    ContainerExpr(ContainerExpr), ArrayBoundExpr(ArrayBoundExpr),
    ContainerNeedsDereference(ContainerNeedsDereference),
    OnlyUsedAsIndex(true),  AliasDecl(NULL), ConfidenceLevel(RL_Safe),
    NextStmtParent(NULL), CurrStmtParent(NULL), ReplaceWithAliasUse(false),
    AliasFromForInit(false) {
     if (ContainerExpr) {
       addComponent(ContainerExpr);
       llvm::FoldingSetNodeID ID;
       const Expr *E = ContainerExpr->IgnoreParenImpCasts();
       E->Profile(ID, *Context, true);
     }
  }

  /// \brief Finds all uses of IndexVar in Body, placing all usages in Usages,
  /// and returns true if IndexVar was only used in a way consistent with a
  /// range-based for loop.
  ///
  /// The general strategy is to reject any DeclRefExprs referencing IndexVar,
  /// with the exception of certain acceptable patterns.
  /// For arrays, the DeclRefExpr for IndexVar must appear as the index of an
  /// ArraySubscriptExpression. Iterator-based loops may dereference
  /// IndexVar or call methods through operator-> (builtin or overloaded).
  /// Array-like containers may use IndexVar as a parameter to the at() member
  /// function and in overloaded operator[].
  bool findAndVerifyUsages(const Stmt *Body) {
    TraverseStmt(const_cast<Stmt *>(Body));
    return OnlyUsedAsIndex && ContainerExpr;
  }

  /// \brief Add a set of components that we should consider relevant to the
  /// container.
  void addComponents(const ComponentVector &Components) {
    // FIXME: add sort(on ID)+unique to avoid extra work.
    for (ComponentVector::const_iterator I = Components.begin(),
                                         E = Components.end(); I != E; ++I)
      addComponent(*I);
  }

  /// \brief Accessor for Usages.
  const UsageResult &getUsages() const { return Usages; }

  /// \brief Get the container indexed by IndexVar, if any.
  const Expr *getContainerIndexed() const {
    return ContainerExpr;
  }

  /// \brief Returns the statement declaring the variable created as an alias
  /// for the loop element, if any.
  const DeclStmt *getAliasDecl() const { return AliasDecl; }

  /// \brief Accessor for ConfidenceLevel.
  RiskLevel getRiskLevel() const {
    return ConfidenceLevel.getRiskLevel();
  }

  /// \brief Indicates if the alias declaration was in a place where it cannot
  /// simply be removed but rather replaced with a use of the alias variable.
  /// For example, variables declared in the condition of an if, switch, or for
  /// stmt.
  bool aliasUseRequired() const { return ReplaceWithAliasUse; }

  /// \brief Indicates if the alias declaration came from the init clause of a
  /// nested for loop. SourceRanges provided by Clang for DeclStmts in this
  /// case need to be adjusted.
  bool aliasFromForInit() const { return AliasFromForInit; }

 private:
  /// Typedef used in CRTP functions.
  typedef RecursiveASTVisitor<ForLoopIndexUseVisitor> VisitorBase;
  friend class RecursiveASTVisitor<ForLoopIndexUseVisitor>;

  /// Overriden methods for RecursiveASTVisitor's traversal.
  bool TraverseArraySubscriptExpr(ArraySubscriptExpr *E);
  bool TraverseCXXMemberCallExpr(CXXMemberCallExpr *MemberCall);
  bool TraverseCXXOperatorCallExpr(CXXOperatorCallExpr *OpCall);
  bool TraverseMemberExpr(MemberExpr *Member);
  bool TraverseUnaryDeref(UnaryOperator *Uop);
  bool VisitDeclRefExpr(DeclRefExpr *E);
  bool VisitDeclStmt(DeclStmt *S);
  bool TraverseStmt(Stmt *S);

  /// \brief Add an expression to the list of expressions on which the container
  /// expression depends.
  void addComponent(const Expr *E) {
    llvm::FoldingSetNodeID ID;
    const Expr *Node = E->IgnoreParenImpCasts();
    Node->Profile(ID, *Context, true);
    DependentExprs.push_back(std::make_pair(Node, ID));
  }

  // Input member variables:
  ASTContext *Context;
  /// The index variable's VarDecl.
  const VarDecl *IndexVar;
  /// The loop's 'end' variable, which cannot be mentioned at all.
  const VarDecl *EndVar;
  /// The Expr which refers to the container.
  const Expr *ContainerExpr;
  /// The Expr which refers to the terminating condition for array-based loops.
  const Expr *ArrayBoundExpr;
  bool ContainerNeedsDereference;

  // Output member variables:
  /// A container which holds all usages of IndexVar as the index of
  /// ArraySubscriptExpressions.
  UsageResult Usages;
  bool OnlyUsedAsIndex;
  /// The DeclStmt for an alias to the container element.
  const DeclStmt *AliasDecl;
  Confidence ConfidenceLevel;
  /// \brief A list of expressions on which ContainerExpr depends.
  ///
  /// If any of these expressions are encountered outside of an acceptable usage
  /// of the loop element, lower our confidence level.
  llvm::SmallVector<
      std::pair<const Expr *, llvm::FoldingSetNodeID>, 16> DependentExprs;

  /// The parent-in-waiting. Will become the real parent once we traverse down
  /// one level in the AST.
  const Stmt *NextStmtParent;
  /// The actual parent of a node when Visit*() calls are made. Only the
  /// parentage of DeclStmt's to possible iteration/selection statements is of
  /// importance.
  const Stmt *CurrStmtParent;

  /// \see aliasUseRequired().
  bool ReplaceWithAliasUse;
  /// \see aliasFromForInit().
  bool AliasFromForInit;
};

/// \brief Obtain the original source code text from a SourceRange.
static StringRef getStringFromRange(SourceManager &SourceMgr,
                                    const LangOptions &LangOpts,
                                    SourceRange Range) {
  if (SourceMgr.getFileID(Range.getBegin()) !=
      SourceMgr.getFileID(Range.getEnd()))
    return NULL;

  CharSourceRange SourceChars(Range, true);
  return Lexer::getSourceText(SourceChars, SourceMgr, LangOpts);
}

/// \brief Returns the DeclRefExpr represented by E, or NULL if there isn't one.
static const DeclRefExpr *getDeclRef(const Expr *E) {
  return dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts());
}

/// \brief If the given expression is actually a DeclRefExpr, find and return
/// the underlying VarDecl; otherwise, return NULL.
static const VarDecl *getReferencedVariable(const Expr *E) {
  if (const DeclRefExpr *DRE = getDeclRef(E))
    return dyn_cast<VarDecl>(DRE->getDecl());
  return NULL;
}

/// \brief Returns true when the given expression is a member expression
/// whose base is `this` (implicitly or not).
static bool isDirectMemberExpr(const Expr *E) {
  if (const MemberExpr *Member = dyn_cast<MemberExpr>(E->IgnoreParenImpCasts()))
    return isa<CXXThisExpr>(Member->getBase()->IgnoreParenImpCasts());
  return false;
}

/// \brief Returns true when two ValueDecls are the same variable.
static bool areSameVariable(const ValueDecl *First, const ValueDecl *Second) {
  return First && Second &&
         First->getCanonicalDecl() == Second->getCanonicalDecl();
}

/// \brief Determines if an expression is a declaration reference to a
/// particular variable.
static bool exprReferencesVariable(const ValueDecl *Target, const Expr *E) {
  if (!Target || !E)
    return false;
  const DeclRefExpr *Decl = getDeclRef(E);
  return Decl && areSameVariable(Target, Decl->getDecl());
}

/// \brief Returns true when two Exprs are equivalent.
static bool areSameExpr(ASTContext *Context, const Expr *First,
                        const Expr *Second) {
  if (!First || !Second)
    return false;

  llvm::FoldingSetNodeID FirstID, SecondID;
  First->Profile(FirstID, *Context, true);
  Second->Profile(SecondID, *Context, true);
  return FirstID == SecondID;
}

/// \brief Look through conversion/copy constructors to find the explicit
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
static const Expr *digThroughConstructors(const Expr *E) {
  if (!E)
    return NULL;
  E = E->IgnoreParenImpCasts();
  if (const CXXConstructExpr *ConstructExpr = dyn_cast<CXXConstructExpr>(E)) {
    // The initial constructor must take exactly one parameter, but base class
    // and deferred constructors can take more.
    if (ConstructExpr->getNumArgs() != 1 ||
        ConstructExpr->getConstructionKind() != CXXConstructExpr::CK_Complete)
      return NULL;
    E = ConstructExpr->getArg(0);
    if (const MaterializeTemporaryExpr *Temp =
        dyn_cast<MaterializeTemporaryExpr>(E))
      E = Temp->GetTemporaryExpr();
    return digThroughConstructors(E);
  }
  return E;
}

/// \brief If the expression is a dereference or call to operator*(), return the
/// operand. Otherwise, return NULL.
static const Expr *getDereferenceOperand(const Expr *E) {
  if (const UnaryOperator *Uop = dyn_cast<UnaryOperator>(E))
    return Uop->getOpcode() == UO_Deref ? Uop->getSubExpr() : NULL;

  if (const CXXOperatorCallExpr *OpCall = dyn_cast<CXXOperatorCallExpr>(E))
    return OpCall->getOperator() == OO_Star && OpCall->getNumArgs() == 1 ?
        OpCall->getArg(0) : NULL;

  return NULL;
}

/// \brief Returns true when the Container contains an Expr equivalent to E.
template<typename ContainerT>
static bool containsExpr(ASTContext *Context, const ContainerT *Container,
                         const Expr *E) {
  llvm::FoldingSetNodeID ID;
  E->Profile(ID, *Context, true);
  for (typename ContainerT::const_iterator I = Container->begin(),
       End = Container->end(); I != End; ++I)
    if (ID == I->second)
      return true;
  return false;
}

/// \brief Returns true when the index expression is a declaration reference to
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
  return Idx && Idx->getType()->isIntegerType()
             && areSameVariable(IndexVar, Idx->getDecl());
}

/// \brief Returns true when the index expression is a declaration reference to
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

/// \brief Returns true when Opcall is a call a one-parameter dereference of
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

/// \brief Returns true when Uop is a dereference of IndexVar.
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

/// \brief Determines whether the given Decl defines a variable initialized to
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
static bool isAliasDecl(const Decl *TheDecl, const VarDecl *IndexVar) {
  const VarDecl *VDecl = dyn_cast<VarDecl>(TheDecl);
  if (!VDecl)
    return false;
  if (!VDecl->hasInit())
    return false;
  const Expr *Init =
      digThroughConstructors(VDecl->getInit()->IgnoreParenImpCasts());
  if (!Init)
    return false;

  switch (Init->getStmtClass()) {
  case Stmt::ArraySubscriptExprClass: {
    const ArraySubscriptExpr *E = cast<ArraySubscriptExpr>(Init);
    // We don't really care which array is used here. We check to make sure
    // it was the correct one later, since the AST will traverse it next.
    return isIndexInSubscriptExpr(E->getIdx(), IndexVar);
  }

  case Stmt::UnaryOperatorClass:
    return isDereferenceOfUop(cast<UnaryOperator>(Init), IndexVar);

  case Stmt::CXXOperatorCallExprClass: {
      const CXXOperatorCallExpr *OpCall = cast<CXXOperatorCallExpr>(Init);
      if (OpCall->getOperator() == OO_Star)
        return isDereferenceOfOpCall(OpCall, IndexVar);
      break;
  }

  default:
    break;
  }
  return false;
}

/// \brief Determines whether the bound of a for loop condition expression is
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
  llvm::APSInt ConditionSize;
  if (!ConditionExpr->isIntegerConstantExpr(ConditionSize, *Context))
    return false;
  llvm::APSInt ArraySize(ConstType->getSize());
  return llvm::APSInt::isSameValue(ConditionSize, ArraySize);
}

/// \brief If the unary operator is a dereference of IndexVar, include it
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
bool ForLoopIndexUseVisitor::TraverseUnaryDeref(UnaryOperator *Uop) {
  // If we dereference an iterator that's actually a pointer, count the
  // occurrence.
  if (isDereferenceOfUop(Uop, IndexVar)) {
    Usages.push_back(Usage(Uop));
    return true;
  }

  return VisitorBase::TraverseUnaryOperator(Uop);
}

/// \brief If the member expression is operator-> (overloaded or not) on
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
  if (const CXXOperatorCallExpr *Call =
      dyn_cast<CXXOperatorCallExpr>(Base->IgnoreParenImpCasts())) {
    // If operator->() is a MemberExpr containing a CXXOperatorCallExpr, then
    // the MemberExpr does not have the expression we want. We therefore catch
    // that instance here.
    // For example, if vector<Foo>::iterator defines operator->(), then the
    // example `i->bar()` at the top of this function is a CXXMemberCallExpr
    // referring to `i->` as the member function called. We want just `i`, so
    // we take the argument to operator->() as the base object.
    if(Call->getOperator() == OO_Arrow) {
      assert(Call->getNumArgs() == 1 &&
             "Operator-> takes more than one argument");
      Obj = getDeclRef(Call->getArg(0));
      ResultExpr = Obj;
      ExprType = Call->getCallReturnType();
    }
  }

  if (Member->isArrow() && Obj && exprReferencesVariable(IndexVar, Obj)) {
    if (ExprType.isNull())
      ExprType = Obj->getType();

    assert(ExprType->isPointerType() && "Operator-> returned non-pointer type");
    // FIXME: This works around not having the location of the arrow operator.
    // Consider adding OperatorLoc to MemberExpr?
    SourceLocation ArrowLoc =
        Lexer::getLocForEndOfToken(Base->getExprLoc(), 0,
                                   Context->getSourceManager(),
                                   Context->getLangOpts());
    // If something complicated is happening (i.e. the next token isn't an
    // arrow), give up on making this work.
    if (!ArrowLoc.isInvalid()) {
      Usages.push_back(Usage(ResultExpr, /*IsArrow=*/true,
                             SourceRange(Base->getExprLoc(), ArrowLoc)));
      return true;
    }
  }
  return TraverseStmt(Member->getBase());
}

/// \brief If a member function call is the at() accessor on the container with
/// IndexVar as the single argument, include it as a valid usage and prune
/// the traversal.
///
/// Member calls on other objects will not be permitted.
/// Calls on the iterator object are not permitted, unless done through
/// operator->(). The one exception is allowing vector::at() for pseudoarrays.
bool ForLoopIndexUseVisitor::TraverseCXXMemberCallExpr(
    CXXMemberCallExpr *MemberCall) {
  MemberExpr *Member =
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
      Usages.push_back(Usage(MemberCall));
      return true;
    }
  }

  if (containsExpr(Context, &DependentExprs, Member->getBase()))
    ConfidenceLevel.lowerTo(RL_Risky);

  return VisitorBase::TraverseCXXMemberCallExpr(MemberCall);
}

/// \brief If an overloaded operator call is a dereference of IndexVar or
/// a subscript of a the container with IndexVar as the single argument,
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
      Usages.push_back(Usage(OpCall));
      return true;
    }
    break;

  case OO_Subscript:
    if (OpCall->getNumArgs() != 2)
      break;
    if (isIndexInSubscriptExpr(Context, OpCall->getArg(1), IndexVar,
                               OpCall->getArg(0), ContainerExpr,
                               ContainerNeedsDereference)) {
      Usages.push_back(Usage(OpCall));
      return true;
    }
    break;

  default:
    break;
  }
  return VisitorBase::TraverseCXXOperatorCallExpr(OpCall);
}

/// \brief If we encounter an array with IndexVar as the index of an
/// ArraySubsriptExpression, note it as a consistent usage and prune the
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
bool ForLoopIndexUseVisitor::TraverseArraySubscriptExpr(
    ArraySubscriptExpr *E) {
  Expr *Arr = E->getBase();
  if (!isIndexInSubscriptExpr(E->getIdx(), IndexVar))
    return VisitorBase::TraverseArraySubscriptExpr(E);

  if ((ContainerExpr && !areSameExpr(Context, Arr->IgnoreParenImpCasts(),
                                     ContainerExpr->IgnoreParenImpCasts()))
      || !arrayMatchesBoundExpr(Context, Arr->IgnoreImpCasts()->getType(),
                                ArrayBoundExpr)) {
    // If we have already discovered the array being indexed and this isn't it
    // or this array doesn't match, mark this loop as unconvertible.
    OnlyUsedAsIndex = false;
    return VisitorBase::TraverseArraySubscriptExpr(E);
  }

  if (!ContainerExpr)
    ContainerExpr = Arr;

  Usages.push_back(Usage(E));
  return true;
}

/// \brief If we encounter a reference to IndexVar in an unpruned branch of the
/// traversal, mark this loop as unconvertible.
///
/// This implements the whitelist for convertible loops: any usages of IndexVar
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
  if (areSameVariable(IndexVar, TheDecl) || areSameVariable(EndVar, TheDecl))
    OnlyUsedAsIndex = false;
  if (containsExpr(Context, &DependentExprs, E))
    ConfidenceLevel.lowerTo(RL_Risky);
  return true;
}

/// \brief If we find that another variable is created just to refer to the loop
/// element, note it for reuse as the loop variable.
///
/// See the comments for isAliasDecl.
bool ForLoopIndexUseVisitor::VisitDeclStmt(DeclStmt *S) {
  if (!AliasDecl && S->isSingleDecl() &&
      isAliasDecl(S->getSingleDecl(), IndexVar)) {
    AliasDecl = S;
    if (CurrStmtParent) {
      if (isa<IfStmt>(CurrStmtParent) ||
          isa<WhileStmt>(CurrStmtParent) ||
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
  // All this pointer swapping is a mechanism for tracking immediate parentage
  // of Stmts.
  const Stmt *OldNextParent = NextStmtParent;
  CurrStmtParent = NextStmtParent;
  NextStmtParent = S;
  bool Result = VisitorBase::TraverseStmt(S);
  NextStmtParent = OldNextParent;
  return Result;
}

//// \brief Apply the source transformations necessary to migrate the loop!
void LoopFixer::doConversion(ASTContext *Context,
                             const VarDecl *IndexVar,
                             const VarDecl *MaybeContainer,
                             StringRef ContainerString,
                             const UsageResult &Usages,
                             const DeclStmt *AliasDecl,
                             bool AliasUseRequired,
                             bool AliasFromForInit,
                             const ForStmt *TheLoop,
                             bool ContainerNeedsDereference,
                             bool DerefByValue,
                             bool DerefByConstRef) {
  std::string VarName;
  bool VarNameFromAlias = Usages.size() == 1 && AliasDecl;
  bool AliasVarIsRef = false;

  if (VarNameFromAlias) {
    const VarDecl *AliasVar = cast<VarDecl>(AliasDecl->getSingleDecl());
    VarName = AliasVar->getName().str();
    AliasVarIsRef = AliasVar->getType()->isReferenceType();

    // We keep along the entire DeclStmt to keep the correct range here.
    const SourceRange &ReplaceRange = AliasDecl->getSourceRange();

    std::string ReplacementText;
    if (AliasUseRequired)
      ReplacementText = VarName;
    else if (AliasFromForInit)
      // FIXME: Clang includes the location of the ';' but only for DeclStmt's
      // in a for loop's init clause. Need to put this ';' back while removing
      // the declaration of the alias variable. This is probably a bug.
      ReplacementText = ";";

    Owner.addReplacementForCurrentTU(Replacement(
        Context->getSourceManager(),
        CharSourceRange::getTokenRange(ReplaceRange), ReplacementText));
    // No further replacements are made to the loop, since the iterator or index
    // was used exactly once - in the initialization of AliasVar.
  } else {
    VariableNamer Namer(&TUInfo.getGeneratedDecls(),
                        &TUInfo.getParentFinder().getStmtToParentStmtMap(),
                        TheLoop, IndexVar, MaybeContainer, Context);
    VarName = Namer.createIndexName();
    // First, replace all usages of the array subscript expression with our new
    // variable.
    for (UsageResult::const_iterator I = Usages.begin(), E = Usages.end();
         I != E; ++I) {
      std::string ReplaceText = I->IsArrow ? VarName + "." : VarName;
      TUInfo.getReplacedVars().insert(std::make_pair(TheLoop, IndexVar));
      Owner.addReplacementForCurrentTU(
          Replacement(Context->getSourceManager(),
                      CharSourceRange::getTokenRange(I->Range), ReplaceText));
    }
  }

  // Now, we need to construct the new range expresion.
  SourceRange ParenRange(TheLoop->getLParenLoc(), TheLoop->getRParenLoc());

  QualType AutoRefType = Context->getAutoDeductType();

  // If the new variable name is from the aliased variable, then the reference
  // type for the new variable should only be used if the aliased variable was
  // declared as a reference.
  if (!VarNameFromAlias || AliasVarIsRef) {
    // If an iterator's operator*() returns a 'T&' we can bind that to 'auto&'.
    // If operator*() returns 'T' we can bind that to 'auto&&' which will deduce
    // to 'T&&'.
    if (DerefByValue)
      AutoRefType = Context->getRValueReferenceType(AutoRefType);
    else {
      if (DerefByConstRef)
        AutoRefType = Context->getConstType(AutoRefType);
      AutoRefType = Context->getLValueReferenceType(AutoRefType);
    }
  }

  std::string MaybeDereference = ContainerNeedsDereference ? "*" : "";
  std::string TypeString = AutoRefType.getAsString();
  std::string Range = ("(" + TypeString + " " + VarName + " : "
                           + MaybeDereference + ContainerString + ")").str();
  Owner.addReplacementForCurrentTU(
      Replacement(Context->getSourceManager(),
                  CharSourceRange::getTokenRange(ParenRange), Range));
  TUInfo.getGeneratedDecls().insert(make_pair(TheLoop, VarName));
}

/// \brief Determine whether Init appears to be an initializing an iterator.
///
/// If it is, returns the object whose begin() or end() method is called, and
/// the output parameter isArrow is set to indicate whether the initialization
/// is called via . or ->.
static const Expr *getContainerFromBeginEndCall(const Expr *Init, bool IsBegin,
                                                bool *IsArrow) {
  // FIXME: Maybe allow declaration/initialization outside of the for loop?
  const CXXMemberCallExpr *TheCall =
      dyn_cast_or_null<CXXMemberCallExpr>(digThroughConstructors(Init));
  if (!TheCall || TheCall->getNumArgs() != 0)
      return NULL;

  const MemberExpr *Member = dyn_cast<MemberExpr>(TheCall->getCallee());
  if (!Member)
    return NULL;
  const std::string Name = Member->getMemberDecl()->getName();
  const std::string TargetName = IsBegin ? "begin" : "end";
  if (Name != TargetName)
    return NULL;

  const Expr *SourceExpr = Member->getBase();
  if (!SourceExpr)
    return NULL;

  *IsArrow = Member->isArrow();
  return SourceExpr;
}

/// \brief Determines the container whose begin() and end() functions are called
/// for an iterator-based loop.
///
/// BeginExpr must be a member call to a function named "begin()", and EndExpr
/// must be a member .
static const Expr *findContainer(ASTContext *Context, const Expr *BeginExpr,
                                 const Expr *EndExpr,
                                 bool *ContainerNeedsDereference) {
  // Now that we know the loop variable and test expression, make sure they are
  // valid.
  bool BeginIsArrow = false;
  bool EndIsArrow = false;
  const Expr *BeginContainerExpr =
      getContainerFromBeginEndCall(BeginExpr, /*IsBegin=*/true, &BeginIsArrow);
  if (!BeginContainerExpr)
      return NULL;

  const Expr *EndContainerExpr =
      getContainerFromBeginEndCall(EndExpr, /*IsBegin=*/false, &EndIsArrow);
  // Disallow loops that try evil things like this (note the dot and arrow):
  //  for (IteratorType It = Obj.begin(), E = Obj->end(); It != E; ++It) { }
  if (!EndContainerExpr || BeginIsArrow != EndIsArrow ||
      !areSameExpr(Context, EndContainerExpr, BeginContainerExpr))
    return NULL;

  *ContainerNeedsDereference = BeginIsArrow;
  return BeginContainerExpr;
}

StringRef LoopFixer::checkDeferralsAndRejections(ASTContext *Context,
                                                 const Expr *ContainerExpr,
                                                 Confidence ConfidenceLevel,
                                                 const ForStmt *TheLoop) {
  // If we already modified the range of this for loop, don't do any further
  // updates on this iteration.
  // FIXME: Once Replacements can detect conflicting edits, replace this
  // implementation and rely on conflicting edit detection instead.
  if (TUInfo.getReplacedVars().count(TheLoop)) {
    ++*DeferredChanges;
    return "";
  }

  TUInfo.getParentFinder().gatherAncestors(Context->getTranslationUnitDecl());
  // Ensure that we do not try to move an expression dependent on a local
  // variable declared inside the loop outside of it!
  DependencyFinderASTVisitor DependencyFinder(
      &TUInfo.getParentFinder().getStmtToParentStmtMap(),
      &TUInfo.getParentFinder().getDeclToParentStmtMap(),
      &TUInfo.getReplacedVars(), TheLoop);

  // Not all of these are actually deferred changes.
  // FIXME: Determine when the external dependency isn't an expression converted
  // by another loop.
  if (DependencyFinder.dependsOnInsideVariable(ContainerExpr)) {
    ++*DeferredChanges;
    return "";
  }
  if (ConfidenceLevel.getRiskLevel() > MaxRisk) {
    ++*RejectedChanges;
    return "";
  }

  StringRef ContainerString;
  if (isa<CXXThisExpr>(ContainerExpr->IgnoreParenImpCasts())) {
    ContainerString = "this";
  } else {
    ContainerString = getStringFromRange(Context->getSourceManager(),
                                         Context->getLangOpts(),
                                         ContainerExpr->getSourceRange());
  }

  // In case someone is using an evil macro, reject this change.
  if (ContainerString.empty())
    ++*RejectedChanges;
  return ContainerString;
}

/// \brief Given that we have verified that the loop's header appears to be
/// convertible, run the complete analysis on the loop to determine if the
/// loop's body is convertible.
void LoopFixer::findAndVerifyUsages(ASTContext *Context,
                                    const VarDecl *LoopVar,
                                    const VarDecl *EndVar,
                                    const Expr *ContainerExpr,
                                    const Expr *BoundExpr,
                                    bool ContainerNeedsDereference,
                                    bool DerefByValue,
                                    bool DerefByConstRef,
                                    const ForStmt *TheLoop,
                                    Confidence ConfidenceLevel) {
  ForLoopIndexUseVisitor Finder(Context, LoopVar, EndVar, ContainerExpr,
                                BoundExpr, ContainerNeedsDereference);
  if (ContainerExpr) {
    ComponentFinderASTVisitor ComponentFinder;
    ComponentFinder.findExprComponents(ContainerExpr->IgnoreParenImpCasts());
    Finder.addComponents(ComponentFinder.getComponents());
  }

  if (!Finder.findAndVerifyUsages(TheLoop->getBody()))
    return;

  ConfidenceLevel.lowerTo(Finder.getRiskLevel());
  if (FixerKind == LFK_Array) {
    // The array being indexed by IndexVar was discovered during traversal.
    ContainerExpr = Finder.getContainerIndexed()->IgnoreParenImpCasts();
    // Very few loops are over expressions that generate arrays rather than
    // array variables. Consider loops over arrays that aren't just represented
    // by a variable to be risky conversions.
    if (!getReferencedVariable(ContainerExpr) &&
        !isDirectMemberExpr(ContainerExpr))
      ConfidenceLevel.lowerTo(RL_Risky);
  }

  std::string ContainerString =
      checkDeferralsAndRejections(Context, ContainerExpr,
                                  ConfidenceLevel, TheLoop);
  if (ContainerString.empty())
    return;

  doConversion(Context, LoopVar, getReferencedVariable(ContainerExpr),
               ContainerString, Finder.getUsages(), Finder.getAliasDecl(),
               Finder.aliasUseRequired(), Finder.aliasFromForInit(), TheLoop,
               ContainerNeedsDereference, DerefByValue, DerefByConstRef);
  ++*AcceptedChanges;
}

/// \brief The LoopFixer callback, which determines if loops discovered by the
/// matchers are convertible, printing information about the loops if so.
void LoopFixer::run(const MatchFinder::MatchResult &Result) {
  const BoundNodes &Nodes = Result.Nodes;
  Confidence ConfidenceLevel(RL_Safe);
  ASTContext *Context = Result.Context;
  const ForStmt *TheLoop = Nodes.getStmtAs<ForStmt>(LoopName);

  if (!Owner.isFileModifiable(Context->getSourceManager(),TheLoop->getForLoc()))
    return;

  // Check that we have exactly one index variable and at most one end variable.
  const VarDecl *LoopVar = Nodes.getDeclAs<VarDecl>(IncrementVarName);
  const VarDecl *CondVar = Nodes.getDeclAs<VarDecl>(ConditionVarName);
  const VarDecl *InitVar = Nodes.getDeclAs<VarDecl>(InitVarName);
  if (!areSameVariable(LoopVar, CondVar) || !areSameVariable(LoopVar, InitVar))
    return;
  const VarDecl *EndVar = Nodes.getDeclAs<VarDecl>(EndVarName);
  const VarDecl *ConditionEndVar =
      Nodes.getDeclAs<VarDecl>(ConditionEndVarName);
  if (EndVar && !areSameVariable(EndVar, ConditionEndVar))
    return;

  // If the end comparison isn't a variable, we can try to work with the
  // expression the loop variable is being tested against instead.
  const CXXMemberCallExpr *EndCall =
      Nodes.getStmtAs<CXXMemberCallExpr>(EndCallName);
  const Expr *BoundExpr = Nodes.getStmtAs<Expr>(ConditionBoundName);
  // If the loop calls end()/size() after each iteration, lower our confidence
  // level.
  if (FixerKind != LFK_Array && !EndVar)
    ConfidenceLevel.lowerTo(RL_Reasonable);

  const Expr *ContainerExpr = NULL;
  bool DerefByValue = false;
  bool DerefByConstRef = false;
  bool ContainerNeedsDereference = false;
  // FIXME: Try to put most of this logic inside a matcher. Currently, matchers
  // don't allow the right-recursive checks in digThroughConstructors.
  if (FixerKind == LFK_Iterator) {
    ContainerExpr = findContainer(Context, LoopVar->getInit(),
                                  EndVar ? EndVar->getInit() : EndCall,
                                  &ContainerNeedsDereference);

    QualType InitVarType = InitVar->getType();
    QualType CanonicalInitVarType = InitVarType.getCanonicalType();

    const CXXMemberCallExpr *BeginCall =
        Nodes.getNodeAs<CXXMemberCallExpr>(BeginCallName);
    assert(BeginCall != 0 && "Bad Callback. No begin call expression.");
    QualType CanonicalBeginType =
        BeginCall->getMethodDecl()->getResultType().getCanonicalType();

    if (CanonicalBeginType->isPointerType() &&
        CanonicalInitVarType->isPointerType()) {
      QualType BeginPointeeType = CanonicalBeginType->getPointeeType();
      QualType InitPointeeType = CanonicalInitVarType->getPointeeType();
      // If the initializer and the variable are both pointers check if the
      // un-qualified pointee types match otherwise we don't use auto.
      if (!Context->hasSameUnqualifiedType(InitPointeeType, BeginPointeeType))
        return;
    } else {
      // Check for qualified types to avoid conversions from non-const to const
      // iterator types.
      if (!Context->hasSameType(CanonicalInitVarType, CanonicalBeginType))
        return;
    }

    DerefByValue = Nodes.getNodeAs<QualType>(DerefByValueResultName) != 0;
    if (!DerefByValue) {
      if (const QualType *DerefType =
              Nodes.getNodeAs<QualType>(DerefByRefResultName)) {
        // A node will only be bound with DerefByRefResultName if we're dealing
        // with a user-defined iterator type. Test the const qualification of
        // the reference type.
        DerefByConstRef = (*DerefType)->getAs<ReferenceType>()->getPointeeType()
            .isConstQualified();
      } else {
        // By nature of the matcher this case is triggered only for built-in
        // iterator types (i.e. pointers).
        assert(isa<PointerType>(CanonicalInitVarType) &&
               "Non-class iterator type is not a pointer type");
        QualType InitPointeeType = CanonicalInitVarType->getPointeeType();
        QualType BeginPointeeType = CanonicalBeginType->getPointeeType();
        // If the initializer and variable have both the same type just use auto
        // otherwise we test for const qualification of the pointed-at type.
        if (!Context->hasSameType(InitPointeeType, BeginPointeeType))
          DerefByConstRef = InitPointeeType.isConstQualified();
      }
    } else {
      // If the dereference operator returns by value then test for the
      // canonical const qualification of the init variable type.
      DerefByConstRef = CanonicalInitVarType.isConstQualified();
    }
  } else if (FixerKind == LFK_PseudoArray) {
    if (!EndCall)
      return;
    ContainerExpr = EndCall->getImplicitObjectArgument();
    const MemberExpr *Member = dyn_cast<MemberExpr>(EndCall->getCallee());
    if (!Member)
      return;
    ContainerNeedsDereference = Member->isArrow();
  }
  // We must know the container or an array length bound.
  if (!ContainerExpr && !BoundExpr)
    return;

  findAndVerifyUsages(Context, LoopVar, EndVar, ContainerExpr, BoundExpr,
                      ContainerNeedsDereference, DerefByValue, DerefByConstRef,
                      TheLoop, ConfidenceLevel);
}
