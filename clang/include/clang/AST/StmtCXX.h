//===--- StmtCXX.h - Classes for representing C++ statements ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the C++ statement AST node classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTCXX_H
#define LLVM_CLANG_AST_STMTCXX_H

#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Stmt.h"
#include "llvm/Support/Compiler.h"

namespace clang {

class VarDecl;
class ExpansionStmtDecl;

/// CXXCatchStmt - This represents a C++ catch block.
///
class CXXCatchStmt : public Stmt {
  SourceLocation CatchLoc;
  /// The exception-declaration of the type.
  VarDecl *ExceptionDecl;
  /// The handler block.
  Stmt *HandlerBlock;

public:
  CXXCatchStmt(SourceLocation catchLoc, VarDecl *exDecl, Stmt *handlerBlock)
  : Stmt(CXXCatchStmtClass), CatchLoc(catchLoc), ExceptionDecl(exDecl),
    HandlerBlock(handlerBlock) {}

  CXXCatchStmt(EmptyShell Empty)
  : Stmt(CXXCatchStmtClass), ExceptionDecl(nullptr), HandlerBlock(nullptr) {}

  SourceLocation getBeginLoc() const LLVM_READONLY { return CatchLoc; }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return HandlerBlock->getEndLoc();
  }

  SourceLocation getCatchLoc() const { return CatchLoc; }
  VarDecl *getExceptionDecl() const { return ExceptionDecl; }
  QualType getCaughtType() const;
  Stmt *getHandlerBlock() const { return HandlerBlock; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXCatchStmtClass;
  }

  child_range children() { return child_range(&HandlerBlock, &HandlerBlock+1); }

  const_child_range children() const {
    return const_child_range(&HandlerBlock, &HandlerBlock + 1);
  }

  friend class ASTStmtReader;
};

/// CXXTryStmt - A C++ try block, including all handlers.
///
class CXXTryStmt final : public Stmt,
                         private llvm::TrailingObjects<CXXTryStmt, Stmt *> {

  friend TrailingObjects;
  friend class ASTStmtReader;

  SourceLocation TryLoc;
  unsigned NumHandlers;
  size_t numTrailingObjects(OverloadToken<Stmt *>) const { return NumHandlers; }

  CXXTryStmt(SourceLocation tryLoc, CompoundStmt *tryBlock,
             ArrayRef<Stmt *> handlers);
  CXXTryStmt(EmptyShell Empty, unsigned numHandlers)
    : Stmt(CXXTryStmtClass), NumHandlers(numHandlers) { }

  Stmt *const *getStmts() const { return getTrailingObjects(); }
  Stmt **getStmts() { return getTrailingObjects(); }

public:
  static CXXTryStmt *Create(const ASTContext &C, SourceLocation tryLoc,
                            CompoundStmt *tryBlock, ArrayRef<Stmt *> handlers);

  static CXXTryStmt *Create(const ASTContext &C, EmptyShell Empty,
                            unsigned numHandlers);

  SourceLocation getBeginLoc() const LLVM_READONLY { return getTryLoc(); }

  SourceLocation getTryLoc() const { return TryLoc; }
  SourceLocation getEndLoc() const {
    return getStmts()[NumHandlers]->getEndLoc();
  }

  CompoundStmt *getTryBlock() {
    return cast<CompoundStmt>(getStmts()[0]);
  }
  const CompoundStmt *getTryBlock() const {
    return cast<CompoundStmt>(getStmts()[0]);
  }

  unsigned getNumHandlers() const { return NumHandlers; }
  CXXCatchStmt *getHandler(unsigned i) {
    return cast<CXXCatchStmt>(getStmts()[i + 1]);
  }
  const CXXCatchStmt *getHandler(unsigned i) const {
    return cast<CXXCatchStmt>(getStmts()[i + 1]);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXTryStmtClass;
  }

  child_range children() {
    return child_range(getStmts(), getStmts() + getNumHandlers() + 1);
  }

  const_child_range children() const {
    return const_child_range(getStmts(), getStmts() + getNumHandlers() + 1);
  }
};

/// CXXForRangeStmt - This represents C++0x [stmt.ranged]'s ranged for
/// statement, represented as 'for (range-declarator : range-expression)'
/// or 'for (init-statement range-declarator : range-expression)'.
///
/// This is stored in a partially-desugared form to allow full semantic
/// analysis of the constituent components. The original syntactic components
/// can be extracted using getLoopVariable and getRangeInit.
class CXXForRangeStmt : public Stmt {
  enum { INIT, RANGE, BEGINSTMT, ENDSTMT, COND, INC, LOOPVAR, BODY, END };
  // SubExprs[RANGE] is an expression or declstmt.
  // SubExprs[COND] and SubExprs[INC] are expressions.
  Stmt *SubExprs[END];
  SourceLocation ForLoc;
  SourceLocation CoawaitLoc;
  SourceLocation ColonLoc;
  SourceLocation RParenLoc;

  friend class ASTStmtReader;
public:
  CXXForRangeStmt(Stmt *InitStmt, DeclStmt *Range, DeclStmt *Begin,
                  DeclStmt *End, Expr *Cond, Expr *Inc, DeclStmt *LoopVar,
                  Stmt *Body, SourceLocation FL, SourceLocation CAL,
                  SourceLocation CL, SourceLocation RPL);
  CXXForRangeStmt(EmptyShell Empty) : Stmt(CXXForRangeStmtClass, Empty) { }

  Stmt *getInit() { return SubExprs[INIT]; }
  VarDecl *getLoopVariable();
  Expr *getRangeInit();

  const Stmt *getInit() const { return SubExprs[INIT]; }
  const VarDecl *getLoopVariable() const;
  const Expr *getRangeInit() const;


  DeclStmt *getRangeStmt() { return cast<DeclStmt>(SubExprs[RANGE]); }
  DeclStmt *getBeginStmt() {
    return cast_or_null<DeclStmt>(SubExprs[BEGINSTMT]);
  }
  DeclStmt *getEndStmt() { return cast_or_null<DeclStmt>(SubExprs[ENDSTMT]); }
  Expr *getCond() { return cast_or_null<Expr>(SubExprs[COND]); }
  Expr *getInc() { return cast_or_null<Expr>(SubExprs[INC]); }
  DeclStmt *getLoopVarStmt() { return cast<DeclStmt>(SubExprs[LOOPVAR]); }
  Stmt *getBody() { return SubExprs[BODY]; }

  const DeclStmt *getRangeStmt() const {
    return cast<DeclStmt>(SubExprs[RANGE]);
  }
  const DeclStmt *getBeginStmt() const {
    return cast_or_null<DeclStmt>(SubExprs[BEGINSTMT]);
  }
  const DeclStmt *getEndStmt() const {
    return cast_or_null<DeclStmt>(SubExprs[ENDSTMT]);
  }
  const Expr *getCond() const {
    return cast_or_null<Expr>(SubExprs[COND]);
  }
  const Expr *getInc() const {
    return cast_or_null<Expr>(SubExprs[INC]);
  }
  const DeclStmt *getLoopVarStmt() const {
    return cast<DeclStmt>(SubExprs[LOOPVAR]);
  }
  const Stmt *getBody() const { return SubExprs[BODY]; }

  void setInit(Stmt *S) { SubExprs[INIT] = S; }
  void setRangeInit(Expr *E) { SubExprs[RANGE] = reinterpret_cast<Stmt*>(E); }
  void setRangeStmt(Stmt *S) { SubExprs[RANGE] = S; }
  void setBeginStmt(Stmt *S) { SubExprs[BEGINSTMT] = S; }
  void setEndStmt(Stmt *S) { SubExprs[ENDSTMT] = S; }
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  void setInc(Expr *E) { SubExprs[INC] = reinterpret_cast<Stmt*>(E); }
  void setLoopVarStmt(Stmt *S) { SubExprs[LOOPVAR] = S; }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getForLoc() const { return ForLoc; }
  SourceLocation getCoawaitLoc() const { return CoawaitLoc; }
  SourceLocation getColonLoc() const { return ColonLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }

  SourceLocation getBeginLoc() const LLVM_READONLY { return ForLoc; }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return SubExprs[BODY]->getEndLoc();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXForRangeStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[END]);
  }

  const_child_range children() const {
    return const_child_range(&SubExprs[0], &SubExprs[END]);
  }
};

/// Representation of a Microsoft __if_exists or __if_not_exists
/// statement with a dependent name.
///
/// The __if_exists statement can be used to include a sequence of statements
/// in the program only when a particular dependent name does not exist. For
/// example:
///
/// \code
/// template<typename T>
/// void call_foo(T &t) {
///   __if_exists (T::foo) {
///     t.foo(); // okay: only called when T::foo exists.
///   }
/// }
/// \endcode
///
/// Similarly, the __if_not_exists statement can be used to include the
/// statements when a particular name does not exist.
///
/// Note that this statement only captures __if_exists and __if_not_exists
/// statements whose name is dependent. All non-dependent cases are handled
/// directly in the parser, so that they don't introduce a new scope. Clang
/// introduces scopes in the dependent case to keep names inside the compound
/// statement from leaking out into the surround statements, which would
/// compromise the template instantiation model. This behavior differs from
/// Visual C++ (which never introduces a scope), but is a fairly reasonable
/// approximation of the VC++ behavior.
class MSDependentExistsStmt : public Stmt {
  SourceLocation KeywordLoc;
  bool IsIfExists;
  NestedNameSpecifierLoc QualifierLoc;
  DeclarationNameInfo NameInfo;
  Stmt *SubStmt;

  friend class ASTReader;
  friend class ASTStmtReader;

public:
  MSDependentExistsStmt(SourceLocation KeywordLoc, bool IsIfExists,
                        NestedNameSpecifierLoc QualifierLoc,
                        DeclarationNameInfo NameInfo,
                        CompoundStmt *SubStmt)
  : Stmt(MSDependentExistsStmtClass),
    KeywordLoc(KeywordLoc), IsIfExists(IsIfExists),
    QualifierLoc(QualifierLoc), NameInfo(NameInfo),
    SubStmt(reinterpret_cast<Stmt *>(SubStmt)) { }

  /// Retrieve the location of the __if_exists or __if_not_exists
  /// keyword.
  SourceLocation getKeywordLoc() const { return KeywordLoc; }

  /// Determine whether this is an __if_exists statement.
  bool isIfExists() const { return IsIfExists; }

  /// Determine whether this is an __if_exists statement.
  bool isIfNotExists() const { return !IsIfExists; }

  /// Retrieve the nested-name-specifier that qualifies this name, if
  /// any.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }

  /// Retrieve the name of the entity we're testing for, along with
  /// location information
  DeclarationNameInfo getNameInfo() const { return NameInfo; }

  /// Retrieve the compound statement that will be included in the
  /// program only if the existence of the symbol matches the initial keyword.
  CompoundStmt *getSubStmt() const {
    return reinterpret_cast<CompoundStmt *>(SubStmt);
  }

  SourceLocation getBeginLoc() const LLVM_READONLY { return KeywordLoc; }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return SubStmt->getEndLoc();
  }

  child_range children() {
    return child_range(&SubStmt, &SubStmt+1);
  }

  const_child_range children() const {
    return const_child_range(&SubStmt, &SubStmt + 1);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == MSDependentExistsStmtClass;
  }
};

/// Represents the body of a coroutine. This wraps the normal function
/// body and holds the additional semantic context required to set up and tear
/// down the coroutine frame.
class CoroutineBodyStmt final
    : public Stmt,
      private llvm::TrailingObjects<CoroutineBodyStmt, Stmt *> {
  enum SubStmt {
    Body,          ///< The body of the coroutine.
    Promise,       ///< The promise statement.
    InitSuspend,   ///< The initial suspend statement, run before the body.
    FinalSuspend,  ///< The final suspend statement, run after the body.
    OnException,   ///< Handler for exceptions thrown in the body.
    OnFallthrough, ///< Handler for control flow falling off the body.
    Allocate,      ///< Coroutine frame memory allocation.
    Deallocate,    ///< Coroutine frame memory deallocation.
    ResultDecl,    ///< Declaration holding the result of get_return_object.
    ReturnValue,   ///< Return value for thunk function: p.get_return_object().
    ReturnStmt,    ///< Return statement for the thunk function.
    ReturnStmtOnAllocFailure, ///< Return statement if allocation failed.
    FirstParamMove ///< First offset for move construction of parameter copies.
  };
  unsigned NumParams;

  friend class ASTStmtReader;
  friend class ASTReader;
  friend TrailingObjects;

  Stmt **getStoredStmts() { return getTrailingObjects(); }

  Stmt *const *getStoredStmts() const { return getTrailingObjects(); }

public:

  struct CtorArgs {
    Stmt *Body = nullptr;
    Stmt *Promise = nullptr;
    Expr *InitialSuspend = nullptr;
    Expr *FinalSuspend = nullptr;
    Stmt *OnException = nullptr;
    Stmt *OnFallthrough = nullptr;
    Expr *Allocate = nullptr;
    Expr *Deallocate = nullptr;
    Stmt *ResultDecl = nullptr;
    Expr *ReturnValue = nullptr;
    Stmt *ReturnStmt = nullptr;
    Stmt *ReturnStmtOnAllocFailure = nullptr;
    ArrayRef<Stmt *> ParamMoves;
  };

private:

  CoroutineBodyStmt(CtorArgs const& Args);

public:
  static CoroutineBodyStmt *Create(const ASTContext &C, CtorArgs const &Args);
  static CoroutineBodyStmt *Create(const ASTContext &C, EmptyShell,
                                   unsigned NumParams);

  bool hasDependentPromiseType() const {
    return getPromiseDecl()->getType()->isDependentType();
  }

  /// Retrieve the body of the coroutine as written. This will be either
  /// a CompoundStmt. If the coroutine is in function-try-block, we will
  /// wrap the CXXTryStmt into a CompoundStmt to keep consistency.
  CompoundStmt *getBody() const {
    return cast<CompoundStmt>(getStoredStmts()[SubStmt::Body]);
  }

  Stmt *getPromiseDeclStmt() const {
    return getStoredStmts()[SubStmt::Promise];
  }
  VarDecl *getPromiseDecl() const {
    return cast<VarDecl>(cast<DeclStmt>(getPromiseDeclStmt())->getSingleDecl());
  }

  Stmt *getInitSuspendStmt() const {
    return getStoredStmts()[SubStmt::InitSuspend];
  }
  Stmt *getFinalSuspendStmt() const {
    return getStoredStmts()[SubStmt::FinalSuspend];
  }

  Stmt *getExceptionHandler() const {
    return getStoredStmts()[SubStmt::OnException];
  }
  Stmt *getFallthroughHandler() const {
    return getStoredStmts()[SubStmt::OnFallthrough];
  }

  Expr *getAllocate() const {
    return cast_or_null<Expr>(getStoredStmts()[SubStmt::Allocate]);
  }
  Expr *getDeallocate() const {
    return cast_or_null<Expr>(getStoredStmts()[SubStmt::Deallocate]);
  }
  Stmt *getResultDecl() const { return getStoredStmts()[SubStmt::ResultDecl]; }
  Expr *getReturnValueInit() const {
    return cast<Expr>(getStoredStmts()[SubStmt::ReturnValue]);
  }
  Expr *getReturnValue() const {
    auto *RS = dyn_cast_or_null<clang::ReturnStmt>(getReturnStmt());
    return RS ? RS->getRetValue() : nullptr;
  }
  Stmt *getReturnStmt() const { return getStoredStmts()[SubStmt::ReturnStmt]; }
  Stmt *getReturnStmtOnAllocFailure() const {
    return getStoredStmts()[SubStmt::ReturnStmtOnAllocFailure];
  }
  ArrayRef<Stmt const *> getParamMoves() const {
    return {getStoredStmts() + SubStmt::FirstParamMove, NumParams};
  }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return getBody() ? getBody()->getBeginLoc()
                     : getPromiseDecl()->getBeginLoc();
  }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return getBody() ? getBody()->getEndLoc() : getPromiseDecl()->getEndLoc();
  }

  child_range children() {
    return child_range(getStoredStmts(),
                       getStoredStmts() + SubStmt::FirstParamMove + NumParams);
  }

  const_child_range children() const {
    return const_child_range(getStoredStmts(), getStoredStmts() +
                                                   SubStmt::FirstParamMove +
                                                   NumParams);
  }

  child_range childrenExclBody() {
    return child_range(getStoredStmts() + SubStmt::Body + 1,
                       getStoredStmts() + SubStmt::FirstParamMove + NumParams);
  }

  const_child_range childrenExclBody() const {
    return const_child_range(getStoredStmts() + SubStmt::Body + 1,
                             getStoredStmts() + SubStmt::FirstParamMove +
                                 NumParams);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CoroutineBodyStmtClass;
  }
};

/// Represents a 'co_return' statement in the C++ Coroutines TS.
///
/// This statament models the initialization of the coroutine promise
/// (encapsulating the eventual notional return value) from an expression
/// (or braced-init-list), followed by termination of the coroutine.
///
/// This initialization is modeled by the evaluation of the operand
/// followed by a call to one of:
///   <promise>.return_value(<operand>)
///   <promise>.return_void()
/// which we name the "promise call".
class CoreturnStmt : public Stmt {
  SourceLocation CoreturnLoc;

  enum SubStmt { Operand, PromiseCall, Count };
  Stmt *SubStmts[SubStmt::Count];

  bool IsImplicit : 1;

  friend class ASTStmtReader;
public:
  CoreturnStmt(SourceLocation CoreturnLoc, Stmt *Operand, Stmt *PromiseCall,
               bool IsImplicit = false)
      : Stmt(CoreturnStmtClass), CoreturnLoc(CoreturnLoc),
        IsImplicit(IsImplicit) {
    SubStmts[SubStmt::Operand] = Operand;
    SubStmts[SubStmt::PromiseCall] = PromiseCall;
  }

  CoreturnStmt(EmptyShell) : CoreturnStmt({}, {}, {}) {}

  SourceLocation getKeywordLoc() const { return CoreturnLoc; }

  /// Retrieve the operand of the 'co_return' statement. Will be nullptr
  /// if none was specified.
  Expr *getOperand() const { return static_cast<Expr*>(SubStmts[Operand]); }

  /// Retrieve the promise call that results from this 'co_return'
  /// statement. Will be nullptr if either the coroutine has not yet been
  /// finalized or the coroutine has no eventual return type.
  Expr *getPromiseCall() const {
    return static_cast<Expr*>(SubStmts[PromiseCall]);
  }

  bool isImplicit() const { return IsImplicit; }
  void setIsImplicit(bool value = true) { IsImplicit = value; }

  SourceLocation getBeginLoc() const LLVM_READONLY { return CoreturnLoc; }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return getOperand() ? getOperand()->getEndLoc() : getBeginLoc();
  }

  child_range children() {
    return child_range(SubStmts, SubStmts + SubStmt::Count);
  }

  const_child_range children() const {
    return const_child_range(SubStmts, SubStmts + SubStmt::Count);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CoreturnStmtClass;
  }
};

/// CXXExpansionStmt - Base class for an unexpanded C++ expansion statement.
///
/// The main purpose for this class is to store the AST nodes common to all
/// variants of expansion statements; it also provides storage for additional
/// subexpressions required by its derived classes. This is to simplify the
/// implementation of 'children()' and friends.
///
/// \see ExpansionStmtDecl
/// \see CXXEnumeratingExpansionStmt
/// \see CXXIteratingExpansionStmt
/// \see CXXDestructuringExpansionStmt
/// \see CXXDependentExpansionStmt
class CXXExpansionStmt : public Stmt {
  friend class ASTStmtReader;

  ExpansionStmtDecl *ParentDecl;
  SourceLocation ForLoc;
  SourceLocation LParenLoc;
  SourceLocation ColonLoc;
  SourceLocation RParenLoc;

protected:
  enum SubStmt {
    INIT,
    VAR,
    BODY,
    FIRST_CHILD_STMT,

    // CXXDependentExpansionStmt
    EXPANSION_INITIALIZER = FIRST_CHILD_STMT,
    COUNT_CXXDependentExpansionStmt,

    // CXXDestructuringExpansionStmt
    DECOMP_DECL = FIRST_CHILD_STMT,
    COUNT_CXXDestructuringExpansionStmt,

    // CXXIteratingExpansionStmt
    RANGE = FIRST_CHILD_STMT,
    BEGIN,
    END,
    COUNT_CXXIteratingExpansionStmt,

    MAX_COUNT = COUNT_CXXIteratingExpansionStmt,
  };

  // Managing the memory for this properly would be rather complicated, and
  // expansion statements are fairly uncommon, so just allocate space for the
  // maximum amount of substatements we could possibly have.
  Stmt *SubStmts[MAX_COUNT];

  CXXExpansionStmt(StmtClass SC, EmptyShell Empty);
  CXXExpansionStmt(StmtClass SC, ExpansionStmtDecl *ESD, Stmt *Init,
                   DeclStmt *ExpansionVar, SourceLocation ForLoc,
                   SourceLocation LParenLoc, SourceLocation ColonLoc,
                   SourceLocation RParenLoc);

public:
  SourceLocation getForLoc() const { return ForLoc; }
  SourceLocation getLParenLoc() const { return LParenLoc; }
  SourceLocation getColonLoc() const { return ColonLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }

  SourceLocation getBeginLoc() const;
  SourceLocation getEndLoc() const {
    return getBody() ? getBody()->getEndLoc() : RParenLoc;
  }

  bool hasDependentSize() const;

  ExpansionStmtDecl *getDecl() { return ParentDecl; }
  const ExpansionStmtDecl *getDecl() const { return ParentDecl; }

  Stmt *getInit() { return SubStmts[INIT]; }
  const Stmt *getInit() const { return SubStmts[INIT]; }
  void setInit(Stmt *S) { SubStmts[INIT] = S; }

  VarDecl *getExpansionVariable();
  const VarDecl *getExpansionVariable() const {
    return const_cast<CXXExpansionStmt *>(this)->getExpansionVariable();
  }

  DeclStmt *getExpansionVarStmt() { return cast<DeclStmt>(SubStmts[VAR]); }
  const DeclStmt *getExpansionVarStmt() const {
    return cast<DeclStmt>(SubStmts[VAR]);
  }

  void setExpansionVarStmt(Stmt *S) { SubStmts[VAR] = S; }

  Stmt *getBody() { return SubStmts[BODY]; }
  const Stmt *getBody() const { return SubStmts[BODY]; }
  void setBody(Stmt *S) { SubStmts[BODY] = S; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() >= firstCXXExpansionStmtConstant &&
           T->getStmtClass() <= lastCXXExpansionStmtConstant;
  }

  child_range children() {
    return child_range(SubStmts, SubStmts + FIRST_CHILD_STMT);
  }

  const_child_range children() const {
    return const_child_range(SubStmts, SubStmts + FIRST_CHILD_STMT);
  }
};

/// Represents an unexpanded enumerating expansion statement.
///
/// An 'enumerating' expansion statement is one whose expansion-initializer
/// is a brace-enclosed expression-list; this list is syntactically similar to
/// an initializer list, but it isn't actually an expression in and of itself
/// (in that it is never evaluated or emitted) and instead is just treated as
/// a group of expressions. The expansion initializer of this is always a
/// 'CXXExpansionInitListExpr'.
///
/// Example:
/// \verbatim
///   template for (auto x : { 1, 2, 3 }) {
///     // ...
///   }
/// \endverbatim
///
/// Note that the expression-list may also contain pack expansions, e.g.
/// '{ 1, xs... }', in which case the expansion size is dependent.
///
/// Here, the '{ 1, 2, 3 }' is parsed as a 'CXXExpansionInitListExpr'. This node
/// handles storing (and pack-expanding) the individual expressions.
///
/// Sema then wraps this with a 'CXXExpansionInitListSelectExpr', which also
/// contains a reference to an integral NTTP that is used as the expansion
/// index; this index is either dependent (if the expansion-size is dependent),
/// or set to a value of I in the I-th expansion during the expansion process.
///
/// The actual expansion is done by 'BuildCXXExpansionInitListSelectExpr()': for
/// example, during the 2nd expansion of '{ a, b, c }', I is equal to 1, and
/// BuildCXXExpansionInitListSelectExpr(), when called via TreeTransform,
/// 'instantiates' the expression '{ a, b, c }' to just 'b'.
class CXXEnumeratingExpansionStmt : public CXXExpansionStmt {
  friend class ASTStmtReader;

public:
  CXXEnumeratingExpansionStmt(EmptyShell Empty);
  CXXEnumeratingExpansionStmt(ExpansionStmtDecl *ESD, Stmt *Init,
                              DeclStmt *ExpansionVar, SourceLocation ForLoc,
                              SourceLocation LParenLoc, SourceLocation ColonLoc,
                              SourceLocation RParenLoc);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXEnumeratingExpansionStmtClass;
  }
};

/// Represents an expansion statement whose expansion-initializer is
/// type-dependent.
///
/// This will be instantiated as either a 'CXXIteratingExpansionStmt' or a
/// 'CXXDestructuringExpansionStmt'. Dependent expansion statements can never
/// be enumerating; those are always stored as a 'CXXEnumeratingExpansionStmt',
/// even if the expansion size is dependent because the expression-list contains
/// a pack.
///
/// Example:
/// \verbatim
///   template <typename T>
///   void f() {
///     template for (auto x : T()) {
///       // ...
///     }
///   }
/// \endverbatim
class CXXDependentExpansionStmt : public CXXExpansionStmt {
  friend class ASTStmtReader;

public:
  CXXDependentExpansionStmt(EmptyShell Empty);
  CXXDependentExpansionStmt(ExpansionStmtDecl *ESD, Stmt *Init,
                            DeclStmt *ExpansionVar, Expr *ExpansionInitializer,
                            SourceLocation ForLoc, SourceLocation LParenLoc,
                            SourceLocation ColonLoc, SourceLocation RParenLoc);

  Expr *getExpansionInitializer() {
    return cast<Expr>(SubStmts[EXPANSION_INITIALIZER]);
  }
  const Expr *getExpansionInitializer() const {
    return cast<Expr>(SubStmts[EXPANSION_INITIALIZER]);
  }
  void setExpansionInitializer(Expr *S) { SubStmts[EXPANSION_INITIALIZER] = S; }

  child_range children() {
    return child_range(SubStmts, SubStmts + COUNT_CXXDependentExpansionStmt);
  }

  const_child_range children() const {
    return const_child_range(SubStmts,
                             SubStmts + COUNT_CXXDependentExpansionStmt);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXDependentExpansionStmtClass;
  }
};

/// Represents an unexpanded iterating expansion statement.
///
/// An 'iterating' expansion statement is one whose expansion-initializer is a
/// a range (i.e. it has a corresponding 'begin()'/'end()' pair that is
/// determined based on a number of conditions as stated in [stmt.expand] and
/// [stmt.ranged]).
///
/// The expression used to compute the size of the expansion is not stored and
/// is only created at the moment of expansion.
///
/// Example:
/// \verbatim
///   static constexpr std::string_view foo = "1234";
///   template for (auto x : foo) {
///     // ...
///   }
/// \endverbatim
class CXXIteratingExpansionStmt : public CXXExpansionStmt {
  friend class ASTStmtReader;

public:
  CXXIteratingExpansionStmt(EmptyShell Empty);
  CXXIteratingExpansionStmt(ExpansionStmtDecl *ESD, Stmt *Init,
                            DeclStmt *ExpansionVar, DeclStmt *Range,
                            DeclStmt *Begin, DeclStmt *End,
                            SourceLocation ForLoc, SourceLocation LParenLoc,
                            SourceLocation ColonLoc, SourceLocation RParenLoc);

  const DeclStmt *getRangeVarStmt() const {
    return cast<DeclStmt>(SubStmts[RANGE]);
  }
  DeclStmt *getRangeVarStmt() { return cast<DeclStmt>(SubStmts[RANGE]); }
  void setRangeVarStmt(DeclStmt *S) { SubStmts[RANGE] = S; }

  const VarDecl *getRangeVar() const {
    return cast<VarDecl>(getRangeVarStmt()->getSingleDecl());
  }

  VarDecl *getRangeVar() {
    return cast<VarDecl>(getRangeVarStmt()->getSingleDecl());
  }

  const DeclStmt *getBeginVarStmt() const {
    return cast<DeclStmt>(SubStmts[BEGIN]);
  }
  DeclStmt *getBeginVarStmt() { return cast<DeclStmt>(SubStmts[BEGIN]); }
  void setBeginVarStmt(DeclStmt *S) { SubStmts[BEGIN] = S; }

  const VarDecl *getBeginVar() const {
    return cast<VarDecl>(getBeginVarStmt()->getSingleDecl());
  }

  VarDecl *getBeginVar() {
    return cast<VarDecl>(getBeginVarStmt()->getSingleDecl());
  }

  const DeclStmt *getEndVarStmt() const {
    return cast<DeclStmt>(SubStmts[END]);
  }
  DeclStmt *getEndVarStmt() { return cast<DeclStmt>(SubStmts[END]); }
  void setEndVarStmt(DeclStmt *S) { SubStmts[END] = S; }

  const VarDecl *getEndVar() const {
    return cast<VarDecl>(getEndVarStmt()->getSingleDecl());
  }

  VarDecl *getEndVar() {
    return cast<VarDecl>(getEndVarStmt()->getSingleDecl());
  }

  child_range children() {
    return child_range(SubStmts, SubStmts + COUNT_CXXIteratingExpansionStmt);
  }

  const_child_range children() const {
    return const_child_range(SubStmts,
                             SubStmts + COUNT_CXXIteratingExpansionStmt);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXIteratingExpansionStmtClass;
  }
};

/// Represents an unexpanded destructuring expansion statement.
///
/// A 'destructuring' expansion statement is any expansion statement that is
/// not enumerating or iterating (i.e. destructuring is the last thing we try,
/// and if it doesn't work, the program is ill-formed).
///
/// This essentially involves treating the expansion-initializer as the
/// initializer of a structured-binding declarations, with the number of
/// bindings and expansion size determined by the usual means (array size,
/// std::tuple_size, etc.).
///
/// Example:
/// \verbatim
///   std::array<int, 3> a {1, 2, 3};
///   template for (auto x : a) {
///     // ...
///   }
/// \endverbatim
///
/// Sema wraps the initializer with a CXXDestructuringExpansionSelectExpr, which
/// selects a binding based on the current expansion index; this is analogous to
/// how 'CXXExpansionInitListSelectExpr' is used; see the documentation of
/// 'CXXEnumeratingExpansionStmt' for more details on this.
class CXXDestructuringExpansionStmt : public CXXExpansionStmt {
  friend class ASTStmtReader;

public:
  CXXDestructuringExpansionStmt(EmptyShell Empty);
  CXXDestructuringExpansionStmt(ExpansionStmtDecl *ESD, Stmt *Init,
                                DeclStmt *ExpansionVar,
                                Stmt *DecompositionDeclStmt,
                                SourceLocation ForLoc, SourceLocation LParenLoc,
                                SourceLocation ColonLoc,
                                SourceLocation RParenLoc);

  Stmt *getDecompositionDeclStmt() { return SubStmts[DECOMP_DECL]; }
  const Stmt *getDecompositionDeclStmt() const { return SubStmts[DECOMP_DECL]; }
  void setDecompositionDeclStmt(Stmt *S) { SubStmts[DECOMP_DECL] = S; }

  DecompositionDecl *getDecompositionDecl();
  const DecompositionDecl *getDecompositionDecl() const {
    return const_cast<CXXDestructuringExpansionStmt *>(this)
        ->getDecompositionDecl();
  }

  child_range children() {
    return child_range(SubStmts,
                       SubStmts + COUNT_CXXDestructuringExpansionStmt);
  }

  const_child_range children() const {
    return const_child_range(SubStmts,
                             SubStmts + COUNT_CXXDestructuringExpansionStmt);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXDestructuringExpansionStmtClass;
  }
};

/// Represents the code generated for an expanded expansion statement.
///
/// This holds 'shared statements' and 'instantiations'; these encode the
/// general underlying pattern that all expansion statements desugar to:
///
/// \verbatim
/// {
///   <shared statements>
///   {
///     <1st instantiation>
///   }
///   ...
///   {
///     <n-th instantiation>
///   }
/// }
/// \endverbatim
///
/// Here, the only thing that is stored in the AST are the 'shared statements'
/// and the 'CompoundStmt's that wrap the 'instantiations'. The outer braces
/// shown above are implicit.
///
/// For example, the CXXExpansionInstantiationStmt that corresponds to the
/// following expansion statement
///
/// \verbatim
///   int a[3]{1, 2, 3};
///   template for (auto x : a) {
///     // ...
///   }
/// \endverbatim
///
/// would be
///
/// \verbatim
/// {
///   auto [__u0, __u1, __u2] = a;
///   {
///     auto x = __u0;
///     // ...
///   }
///   {
///     auto x = __u1;
///     // ...
///   }
///   {
///     auto x = __u2;
///     // ...
///   }
/// }
/// \endverbatim
class CXXExpansionInstantiationStmt final
    : public Stmt,
      llvm::TrailingObjects<CXXExpansionInstantiationStmt, Stmt *> {
  friend class ASTStmtReader;
  friend TrailingObjects;

  SourceLocation BeginLoc;
  SourceLocation EndLoc;

  // Instantiations are stored first, then shared statements.
  const unsigned NumInstantiations : 20;
  const unsigned NumSharedStmts : 3;
  unsigned ShouldApplyLifetimeExtensionToSharedStmts : 1;

  CXXExpansionInstantiationStmt(EmptyShell Empty, unsigned NumInstantiations,
                                unsigned NumSharedStmts);
  CXXExpansionInstantiationStmt(SourceLocation BeginLoc, SourceLocation EndLoc,
                                ArrayRef<Stmt *> Instantiations,
                                ArrayRef<Stmt *> SharedStmts,
                                bool ShouldApplyLifetimeExtensionToSharedStmts);

public:
  static CXXExpansionInstantiationStmt *
  Create(ASTContext &C, SourceLocation BeginLoc, SourceLocation EndLoc,
         ArrayRef<Stmt *> Instantiations, ArrayRef<Stmt *> SharedStmts,
         bool ShouldApplyLifetimeExtensionToSharedStmts);

  static CXXExpansionInstantiationStmt *CreateEmpty(ASTContext &C,
                                                    EmptyShell Empty,
                                                    unsigned NumInstantiations,
                                                    unsigned NumSharedStmts);

  ArrayRef<Stmt *> getAllSubStmts() const {
    return getTrailingObjects(getNumSubStmts());
  }

  MutableArrayRef<Stmt *> getAllSubStmts() {
    return getTrailingObjects(getNumSubStmts());
  }

  unsigned getNumSubStmts() const { return NumInstantiations + NumSharedStmts; }

  ArrayRef<Stmt *> getInstantiations() const {
    return getTrailingObjects(NumInstantiations);
  }

  ArrayRef<Stmt *> getSharedStmts() const {
    return getAllSubStmts().drop_front(NumInstantiations);
  }

  bool shouldApplyLifetimeExtensionToSharedStmts() const {
    return ShouldApplyLifetimeExtensionToSharedStmts;
  }

  void setShouldApplyLifetimeExtensionToSharedStmts(bool Apply) {
    ShouldApplyLifetimeExtensionToSharedStmts = Apply;
  }

  SourceLocation getBeginLoc() const { return BeginLoc; }
  SourceLocation getEndLoc() const { return EndLoc; }

  child_range children() {
    Stmt **S = getTrailingObjects();
    return child_range(S, S + getNumSubStmts());
  }

  const_child_range children() const {
    Stmt *const *S = getTrailingObjects();
    return const_child_range(S, S + getNumSubStmts());
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXExpansionInstantiationStmtClass;
  }
};

}  // end namespace clang

#endif
