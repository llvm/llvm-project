//===- StmtOpenACC.h - Classes for OpenACC directives  ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines OpenACC AST classes for statement-level contructs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTOPENACC_H
#define LLVM_CLANG_AST_STMTOPENACC_H

#include "clang/AST/OpenACCClause.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/OpenACCKinds.h"
#include "clang/Basic/SourceLocation.h"
#include <memory>

namespace clang {
/// This is the base class for an OpenACC statement-level construct, other
/// construct types are expected to inherit from this.
class OpenACCConstructStmt : public Stmt {
  friend class ASTStmtWriter;
  friend class ASTStmtReader;
  /// The directive kind. Each implementation of this interface should handle
  /// specific kinds.
  OpenACCDirectiveKind Kind = OpenACCDirectiveKind::Invalid;
  /// The location of the directive statement, from the '#' to the last token of
  /// the directive.
  SourceRange Range;
  /// The location of the directive name.
  SourceLocation DirectiveLoc;

  /// The list of clauses.  This is stored here as an ArrayRef, as this is the
  /// most convienient place to access the list, however the list itself should
  /// be stored in leaf nodes, likely in trailing-storage.
  MutableArrayRef<const OpenACCClause *> Clauses;

protected:
  OpenACCConstructStmt(StmtClass SC, OpenACCDirectiveKind K,
                       SourceLocation Start, SourceLocation DirectiveLoc,
                       SourceLocation End)
      : Stmt(SC), Kind(K), Range(Start, End), DirectiveLoc(DirectiveLoc) {}

  // Used only for initialization, the leaf class can initialize this to
  // trailing storage.
  void setClauseList(MutableArrayRef<const OpenACCClause *> NewClauses) {
    assert(Clauses.empty() && "Cannot change clause list");
    Clauses = NewClauses;
  }

public:
  OpenACCDirectiveKind getDirectiveKind() const { return Kind; }

  static bool classof(const Stmt *S) {
    return S->getStmtClass() >= firstOpenACCConstructStmtConstant &&
           S->getStmtClass() <= lastOpenACCConstructStmtConstant;
  }

  SourceLocation getBeginLoc() const { return Range.getBegin(); }
  SourceLocation getEndLoc() const { return Range.getEnd(); }
  SourceLocation getDirectiveLoc() const { return DirectiveLoc; }
  ArrayRef<const OpenACCClause *> clauses() const { return Clauses; }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_cast<OpenACCConstructStmt *>(this)->children();
  }
};

/// This is a base class for any OpenACC statement-level constructs that have an
/// associated statement. This class is not intended to be instantiated, but is
/// a convenient place to hold the associated statement.
class OpenACCAssociatedStmtConstruct : public OpenACCConstructStmt {
  friend class ASTStmtWriter;
  friend class ASTStmtReader;
  template <typename Derived> friend class RecursiveASTVisitor;
  Stmt *AssociatedStmt = nullptr;

protected:
  OpenACCAssociatedStmtConstruct(StmtClass SC, OpenACCDirectiveKind K,
                                 SourceLocation Start,
                                 SourceLocation DirectiveLoc,
                                 SourceLocation End, Stmt *AssocStmt)
      : OpenACCConstructStmt(SC, K, Start, DirectiveLoc, End),
        AssociatedStmt(AssocStmt) {}

  void setAssociatedStmt(Stmt *S) { AssociatedStmt = S; }
  Stmt *getAssociatedStmt() { return AssociatedStmt; }
  const Stmt *getAssociatedStmt() const {
    return const_cast<OpenACCAssociatedStmtConstruct *>(this)
        ->getAssociatedStmt();
  }

public:
  static bool classof(const Stmt *T) {
    return false;
  }

  child_range children() {
    if (getAssociatedStmt())
      return child_range(&AssociatedStmt, &AssociatedStmt + 1);
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_cast<OpenACCAssociatedStmtConstruct *>(this)->children();
  }
};

/// This class represents a compute construct, representing a 'Kind' of
/// `parallel', 'serial', or 'kernel'. These constructs are associated with a
/// 'structured block', defined as:
///
///  in C or C++, an executable statement, possibly compound, with a single
///  entry at the top and a single exit at the bottom
///
/// At the moment there is no real motivation to have a different AST node for
/// those three, as they are semantically identical, and have only minor
/// differences in the permitted list of clauses, which can be differentiated by
/// the 'Kind'.
class OpenACCComputeConstruct final
    : public OpenACCAssociatedStmtConstruct,
      private llvm::TrailingObjects<OpenACCComputeConstruct,
                                    const OpenACCClause *> {
  friend class ASTStmtWriter;
  friend class ASTStmtReader;
  friend class ASTContext;
  friend TrailingObjects;
  OpenACCComputeConstruct(unsigned NumClauses)
      : OpenACCAssociatedStmtConstruct(
            OpenACCComputeConstructClass, OpenACCDirectiveKind::Invalid,
            SourceLocation{}, SourceLocation{}, SourceLocation{},
            /*AssociatedStmt=*/nullptr) {
    // We cannot send the TrailingObjects storage to the base class (which holds
    // a reference to the data) until it is constructed, so we have to set it
    // separately here.
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }

  OpenACCComputeConstruct(OpenACCDirectiveKind K, SourceLocation Start,
                          SourceLocation DirectiveLoc, SourceLocation End,
                          ArrayRef<const OpenACCClause *> Clauses,
                          Stmt *StructuredBlock)
      : OpenACCAssociatedStmtConstruct(OpenACCComputeConstructClass, K, Start,
                                       DirectiveLoc, End, StructuredBlock) {
    assert(isOpenACCComputeDirectiveKind(K) &&
           "Only parallel, serial, and kernels constructs should be "
           "represented by this type");

    // Initialize the trailing storage.
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());

    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }

  void setStructuredBlock(Stmt *S) { setAssociatedStmt(S); }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCComputeConstructClass;
  }

  static OpenACCComputeConstruct *CreateEmpty(const ASTContext &C,
                                              unsigned NumClauses);
  static OpenACCComputeConstruct *
  Create(const ASTContext &C, OpenACCDirectiveKind K, SourceLocation BeginLoc,
         SourceLocation DirectiveLoc, SourceLocation EndLoc,
         ArrayRef<const OpenACCClause *> Clauses, Stmt *StructuredBlock);

  Stmt *getStructuredBlock() { return getAssociatedStmt(); }
  const Stmt *getStructuredBlock() const {
    return const_cast<OpenACCComputeConstruct *>(this)->getStructuredBlock();
  }
};
/// This class represents a 'loop' construct.  The 'loop' construct applies to a
/// 'for' loop (or range-for loop), and is optionally associated with a Compute
/// Construct.
class OpenACCLoopConstruct final
    : public OpenACCAssociatedStmtConstruct,
      private llvm::TrailingObjects<OpenACCLoopConstruct,
                                   const OpenACCClause *> {
  // The compute/combined construct kind this loop is associated with, or
  // invalid if this is an orphaned loop construct.
  OpenACCDirectiveKind ParentComputeConstructKind =
      OpenACCDirectiveKind::Invalid;

  friend class ASTStmtWriter;
  friend class ASTStmtReader;
  friend class ASTContext;
  friend class OpenACCAssociatedStmtConstruct;
  friend class OpenACCCombinedConstruct;
  friend class OpenACCComputeConstruct;
  friend TrailingObjects;

  OpenACCLoopConstruct(unsigned NumClauses);

  OpenACCLoopConstruct(OpenACCDirectiveKind ParentKind, SourceLocation Start,
                       SourceLocation DirLoc, SourceLocation End,
                       ArrayRef<const OpenACCClause *> Clauses, Stmt *Loop);

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCLoopConstructClass;
  }

  static OpenACCLoopConstruct *CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses);

  static OpenACCLoopConstruct *
  Create(const ASTContext &C, OpenACCDirectiveKind ParentKind,
         SourceLocation BeginLoc, SourceLocation DirLoc, SourceLocation EndLoc,
         ArrayRef<const OpenACCClause *> Clauses, Stmt *Loop);

  Stmt *getLoop() { return getAssociatedStmt(); }
  const Stmt *getLoop() const {
    return const_cast<OpenACCLoopConstruct *>(this)->getLoop();
  }

  /// OpenACC 3.3 2.9:
  /// An orphaned loop construct is a loop construct that is not lexically
  /// enclosed within a compute construct. The parent compute construct of a
  /// loop construct is the nearest compute construct that lexically contains
  /// the loop construct.
  bool isOrphanedLoopConstruct() const {
    return ParentComputeConstructKind == OpenACCDirectiveKind::Invalid;
  }

  OpenACCDirectiveKind getParentComputeConstructKind() const {
    return ParentComputeConstructKind;
  }
};

// This class represents a 'combined' construct, which has a bunch of rules
// shared with both loop and compute constructs.
class OpenACCCombinedConstruct final
    : public OpenACCAssociatedStmtConstruct,
      private llvm::TrailingObjects<OpenACCCombinedConstruct,
                                   const OpenACCClause *> {
  friend TrailingObjects;
  OpenACCCombinedConstruct(unsigned NumClauses)
      : OpenACCAssociatedStmtConstruct(
            OpenACCCombinedConstructClass, OpenACCDirectiveKind::Invalid,
            SourceLocation{}, SourceLocation{}, SourceLocation{},
            /*AssociatedStmt=*/nullptr) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }

  OpenACCCombinedConstruct(OpenACCDirectiveKind K, SourceLocation Start,
                           SourceLocation DirectiveLoc, SourceLocation End,
                           ArrayRef<const OpenACCClause *> Clauses,
                           Stmt *StructuredBlock)
      : OpenACCAssociatedStmtConstruct(OpenACCCombinedConstructClass, K, Start,
                                       DirectiveLoc, End, StructuredBlock) {
    assert(isOpenACCCombinedDirectiveKind(K) &&
           "Only parallel loop, serial loop, and kernels loop constructs "
           "should be represented by this type");

    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }
  void setStructuredBlock(Stmt *S) { setAssociatedStmt(S); }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCCombinedConstructClass;
  }

  static OpenACCCombinedConstruct *CreateEmpty(const ASTContext &C,
                                               unsigned NumClauses);
  static OpenACCCombinedConstruct *
  Create(const ASTContext &C, OpenACCDirectiveKind K, SourceLocation Start,
         SourceLocation DirectiveLoc, SourceLocation End,
         ArrayRef<const OpenACCClause *> Clauses, Stmt *StructuredBlock);
  Stmt *getLoop() { return getAssociatedStmt(); }
  const Stmt *getLoop() const {
    return const_cast<OpenACCCombinedConstruct *>(this)->getLoop();
  }
};

// This class represents a 'data' construct, which has an associated statement
// and clauses, but is otherwise pretty simple.
class OpenACCDataConstruct final
    : public OpenACCAssociatedStmtConstruct,
      private llvm::TrailingObjects<OpenACCDataConstruct,
                                   const OpenACCClause *> {
  friend TrailingObjects;
  OpenACCDataConstruct(unsigned NumClauses)
      : OpenACCAssociatedStmtConstruct(
            OpenACCDataConstructClass, OpenACCDirectiveKind::Data,
            SourceLocation{}, SourceLocation{}, SourceLocation{},
            /*AssociatedStmt=*/nullptr) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }

  OpenACCDataConstruct(SourceLocation Start, SourceLocation DirectiveLoc,
                       SourceLocation End,
                       ArrayRef<const OpenACCClause *> Clauses,
                       Stmt *StructuredBlock)
      : OpenACCAssociatedStmtConstruct(OpenACCDataConstructClass,
                                       OpenACCDirectiveKind::Data, Start,
                                       DirectiveLoc, End, StructuredBlock) {
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }
  void setStructuredBlock(Stmt *S) { setAssociatedStmt(S); }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCDataConstructClass;
  }

  static OpenACCDataConstruct *CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses);
  static OpenACCDataConstruct *Create(const ASTContext &C, SourceLocation Start,
                                      SourceLocation DirectiveLoc,
                                      SourceLocation End,
                                      ArrayRef<const OpenACCClause *> Clauses,
                                      Stmt *StructuredBlock);
  Stmt *getStructuredBlock() { return getAssociatedStmt(); }
  const Stmt *getStructuredBlock() const {
    return const_cast<OpenACCDataConstruct *>(this)->getStructuredBlock();
  }
};
// This class represents a 'enter data' construct, which JUST has clauses.
class OpenACCEnterDataConstruct final
    : public OpenACCConstructStmt,
      private llvm::TrailingObjects<OpenACCEnterDataConstruct,
                                   const OpenACCClause *> {
  friend TrailingObjects;
  OpenACCEnterDataConstruct(unsigned NumClauses)
      : OpenACCConstructStmt(OpenACCEnterDataConstructClass,
                             OpenACCDirectiveKind::EnterData, SourceLocation{},
                             SourceLocation{}, SourceLocation{}) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }
  OpenACCEnterDataConstruct(SourceLocation Start, SourceLocation DirectiveLoc,
                            SourceLocation End,
                            ArrayRef<const OpenACCClause *> Clauses)
      : OpenACCConstructStmt(OpenACCEnterDataConstructClass,
                             OpenACCDirectiveKind::EnterData, Start,
                             DirectiveLoc, End) {
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCEnterDataConstructClass;
  }
  static OpenACCEnterDataConstruct *CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses);
  static OpenACCEnterDataConstruct *
  Create(const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
         SourceLocation End, ArrayRef<const OpenACCClause *> Clauses);
};
// This class represents a 'exit data' construct, which JUST has clauses.
class OpenACCExitDataConstruct final
    : public OpenACCConstructStmt,
      private llvm::TrailingObjects<OpenACCExitDataConstruct,
                                   const OpenACCClause *> {
  friend TrailingObjects;
  OpenACCExitDataConstruct(unsigned NumClauses)
      : OpenACCConstructStmt(OpenACCExitDataConstructClass,
                             OpenACCDirectiveKind::ExitData, SourceLocation{},
                             SourceLocation{}, SourceLocation{}) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }
  OpenACCExitDataConstruct(SourceLocation Start, SourceLocation DirectiveLoc,
                           SourceLocation End,
                           ArrayRef<const OpenACCClause *> Clauses)
      : OpenACCConstructStmt(OpenACCExitDataConstructClass,
                             OpenACCDirectiveKind::ExitData, Start,
                             DirectiveLoc, End) {
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCExitDataConstructClass;
  }
  static OpenACCExitDataConstruct *CreateEmpty(const ASTContext &C,
                                               unsigned NumClauses);
  static OpenACCExitDataConstruct *
  Create(const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
         SourceLocation End, ArrayRef<const OpenACCClause *> Clauses);
};
// This class represents a 'host_data' construct, which has an associated
// statement and clauses, but is otherwise pretty simple.
class OpenACCHostDataConstruct final
    : public OpenACCAssociatedStmtConstruct,
      private llvm::TrailingObjects<OpenACCHostDataConstruct,
                                   const OpenACCClause *> {
  friend TrailingObjects;
  OpenACCHostDataConstruct(unsigned NumClauses)
      : OpenACCAssociatedStmtConstruct(
            OpenACCHostDataConstructClass, OpenACCDirectiveKind::HostData,
            SourceLocation{}, SourceLocation{}, SourceLocation{},
            /*AssociatedStmt=*/nullptr) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }
  OpenACCHostDataConstruct(SourceLocation Start, SourceLocation DirectiveLoc,
                           SourceLocation End,
                           ArrayRef<const OpenACCClause *> Clauses,
                           Stmt *StructuredBlock)
      : OpenACCAssociatedStmtConstruct(OpenACCHostDataConstructClass,
                                       OpenACCDirectiveKind::HostData, Start,
                                       DirectiveLoc, End, StructuredBlock) {
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }
  void setStructuredBlock(Stmt *S) { setAssociatedStmt(S); }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCHostDataConstructClass;
  }
  static OpenACCHostDataConstruct *CreateEmpty(const ASTContext &C,
                                               unsigned NumClauses);
  static OpenACCHostDataConstruct *
  Create(const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
         SourceLocation End, ArrayRef<const OpenACCClause *> Clauses,
         Stmt *StructuredBlock);
  Stmt *getStructuredBlock() { return getAssociatedStmt(); }
  const Stmt *getStructuredBlock() const {
    return const_cast<OpenACCHostDataConstruct *>(this)->getStructuredBlock();
  }
};

// This class represents a 'wait' construct, which has some expressions plus a
// clause list.
class OpenACCWaitConstruct final
    : public OpenACCConstructStmt,
      private llvm::TrailingObjects<OpenACCWaitConstruct, Expr *,
                                    OpenACCClause *> {
  // FIXME: We should be storing a `const OpenACCClause *` to be consistent with
  // the rest of the constructs, but TrailingObjects doesn't allow for mixing
  // constness in its implementation of `getTrailingObjects`.

  friend TrailingObjects;
  friend class ASTStmtWriter;
  friend class ASTStmtReader;
  // Locations of the left and right parens of the 'wait-argument'
  // expression-list.
  SourceLocation LParenLoc, RParenLoc;
  // Location of the 'queues' keyword, if present.
  SourceLocation QueuesLoc;

  // Number of the expressions being represented.  Index '0' is always the
  // 'devnum' expression, even if it not present.
  unsigned NumExprs = 0;

  OpenACCWaitConstruct(unsigned NumExprs, unsigned NumClauses)
      : OpenACCConstructStmt(OpenACCWaitConstructClass,
                             OpenACCDirectiveKind::Wait, SourceLocation{},
                             SourceLocation{}, SourceLocation{}),
        NumExprs(NumExprs) {
    assert(NumExprs >= 1 &&
           "NumExprs should always be >= 1 because the 'devnum' "
           "expr is represented by a null if necessary");
    std::uninitialized_value_construct(getExprPtr(),
                                       getExprPtr() + NumExprs);
    std::uninitialized_value_construct(getTrailingObjects<OpenACCClause *>(),
                                       getTrailingObjects<OpenACCClause *>() +
                                           NumClauses);
    setClauseList(MutableArrayRef(const_cast<const OpenACCClause **>(
                                      getTrailingObjects<OpenACCClause *>()),
                                  NumClauses));
  }

  OpenACCWaitConstruct(SourceLocation Start, SourceLocation DirectiveLoc,
                       SourceLocation LParenLoc, Expr *DevNumExpr,
                       SourceLocation QueuesLoc, ArrayRef<Expr *> QueueIdExprs,
                       SourceLocation RParenLoc, SourceLocation End,
                       ArrayRef<const OpenACCClause *> Clauses)
      : OpenACCConstructStmt(OpenACCWaitConstructClass,
                             OpenACCDirectiveKind::Wait, Start, DirectiveLoc,
                             End),
        LParenLoc(LParenLoc), RParenLoc(RParenLoc), QueuesLoc(QueuesLoc),
        NumExprs(QueueIdExprs.size() + 1) {
    assert(NumExprs >= 1 &&
           "NumExprs should always be >= 1 because the 'devnum' "
           "expr is represented by a null if necessary");

    std::uninitialized_copy(&DevNumExpr, &DevNumExpr + 1,
                            getExprPtr());
    std::uninitialized_copy(QueueIdExprs.begin(), QueueIdExprs.end(),
                            getExprPtr() + 1);

    std::uninitialized_copy(const_cast<OpenACCClause **>(Clauses.begin()),
                            const_cast<OpenACCClause **>(Clauses.end()),
                            getTrailingObjects<OpenACCClause *>());
    setClauseList(MutableArrayRef(const_cast<const OpenACCClause **>(
                                      getTrailingObjects<OpenACCClause *>()),
                                  Clauses.size()));
  }

  size_t numTrailingObjects(OverloadToken<Expr *>) const { return NumExprs; }
  size_t numTrailingObjects(OverloadToken<const OpenACCClause *>) const {
    return clauses().size();
  }

  Expr **getExprPtr() const {
    return const_cast<Expr**>(getTrailingObjects<Expr *>());
  }

  llvm::ArrayRef<Expr *> getExprs() const {
    return llvm::ArrayRef<Expr *>(getExprPtr(), NumExprs);
  }

  llvm::ArrayRef<Expr *> getExprs() {
    return llvm::ArrayRef<Expr *>(getExprPtr(), NumExprs);
  }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCWaitConstructClass;
  }

  static OpenACCWaitConstruct *
  CreateEmpty(const ASTContext &C, unsigned NumExprs, unsigned NumClauses);

  static OpenACCWaitConstruct *
  Create(const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
         SourceLocation LParenLoc, Expr *DevNumExpr, SourceLocation QueuesLoc,
         ArrayRef<Expr *> QueueIdExprs, SourceLocation RParenLoc,
         SourceLocation End, ArrayRef<const OpenACCClause *> Clauses);

  SourceLocation getLParenLoc() const { return LParenLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  bool hasQueuesTag() const { return !QueuesLoc.isInvalid(); }
  SourceLocation getQueuesLoc() const { return QueuesLoc; }

  bool hasDevNumExpr() const { return getExprs()[0]; }
  Expr *getDevNumExpr() const { return getExprs()[0]; }
  llvm::ArrayRef<Expr *> getQueueIdExprs() { return getExprs().drop_front(); }
  llvm::ArrayRef<Expr *> getQueueIdExprs() const {
    return getExprs().drop_front();
  }

  child_range children() {
    Stmt **Begin = reinterpret_cast<Stmt **>(getExprPtr());
    return child_range(Begin, Begin + NumExprs);
  }

  const_child_range children() const {
    Stmt *const *Begin =
        reinterpret_cast<Stmt *const *>(getExprPtr());
    return const_child_range(Begin, Begin + NumExprs);
  }
};

// This class represents an 'init' construct, which has just a clause list.
class OpenACCInitConstruct final
    : public OpenACCConstructStmt,
      private llvm::TrailingObjects<OpenACCInitConstruct,
                                    const OpenACCClause *> {
  friend TrailingObjects;
  OpenACCInitConstruct(unsigned NumClauses)
      : OpenACCConstructStmt(OpenACCInitConstructClass,
                             OpenACCDirectiveKind::Init, SourceLocation{},
                             SourceLocation{}, SourceLocation{}) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }
  OpenACCInitConstruct(SourceLocation Start, SourceLocation DirectiveLoc,
                       SourceLocation End,
                       ArrayRef<const OpenACCClause *> Clauses)
      : OpenACCConstructStmt(OpenACCInitConstructClass,
                             OpenACCDirectiveKind::Init, Start, DirectiveLoc,
                             End) {
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCInitConstructClass;
  }
  static OpenACCInitConstruct *CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses);
  static OpenACCInitConstruct *Create(const ASTContext &C, SourceLocation Start,
                                      SourceLocation DirectiveLoc,
                                      SourceLocation End,
                                      ArrayRef<const OpenACCClause *> Clauses);
};

// This class represents a 'shutdown' construct, which has just a clause list.
class OpenACCShutdownConstruct final
    : public OpenACCConstructStmt,
      private llvm::TrailingObjects<OpenACCShutdownConstruct,
                                    const OpenACCClause *> {
  friend TrailingObjects;
  OpenACCShutdownConstruct(unsigned NumClauses)
      : OpenACCConstructStmt(OpenACCShutdownConstructClass,
                             OpenACCDirectiveKind::Shutdown, SourceLocation{},
                             SourceLocation{}, SourceLocation{}) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }
  OpenACCShutdownConstruct(SourceLocation Start, SourceLocation DirectiveLoc,
                           SourceLocation End,
                           ArrayRef<const OpenACCClause *> Clauses)
      : OpenACCConstructStmt(OpenACCShutdownConstructClass,
                             OpenACCDirectiveKind::Shutdown, Start,
                             DirectiveLoc, End) {
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCShutdownConstructClass;
  }
  static OpenACCShutdownConstruct *CreateEmpty(const ASTContext &C,
                                               unsigned NumClauses);
  static OpenACCShutdownConstruct *
  Create(const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
         SourceLocation End, ArrayRef<const OpenACCClause *> Clauses);
};

// This class represents a 'set' construct, which has just a clause list.
class OpenACCSetConstruct final
    : public OpenACCConstructStmt,
      private llvm::TrailingObjects<OpenACCSetConstruct,
                                    const OpenACCClause *> {
  friend TrailingObjects;
  OpenACCSetConstruct(unsigned NumClauses)
      : OpenACCConstructStmt(OpenACCSetConstructClass,
                             OpenACCDirectiveKind::Set, SourceLocation{},
                             SourceLocation{}, SourceLocation{}) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }

  OpenACCSetConstruct(SourceLocation Start, SourceLocation DirectiveLoc,
                      SourceLocation End,
                      ArrayRef<const OpenACCClause *> Clauses)
      : OpenACCConstructStmt(OpenACCSetConstructClass,
                             OpenACCDirectiveKind::Set, Start, DirectiveLoc,
                             End) {
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCSetConstructClass;
  }
  static OpenACCSetConstruct *CreateEmpty(const ASTContext &C,
                                          unsigned NumClauses);
  static OpenACCSetConstruct *Create(const ASTContext &C, SourceLocation Start,
                                     SourceLocation DirectiveLoc,
                                     SourceLocation End,
                                     ArrayRef<const OpenACCClause *> Clauses);
};
// This class represents an 'update' construct, which has just a clause list.
class OpenACCUpdateConstruct final
    : public OpenACCConstructStmt,
      private llvm::TrailingObjects<OpenACCUpdateConstruct,
                                    const OpenACCClause *> {
  friend TrailingObjects;
  OpenACCUpdateConstruct(unsigned NumClauses)
      : OpenACCConstructStmt(OpenACCUpdateConstructClass,
                             OpenACCDirectiveKind::Update, SourceLocation{},
                             SourceLocation{}, SourceLocation{}) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }

  OpenACCUpdateConstruct(SourceLocation Start, SourceLocation DirectiveLoc,
                         SourceLocation End,
                         ArrayRef<const OpenACCClause *> Clauses)
      : OpenACCConstructStmt(OpenACCUpdateConstructClass,
                             OpenACCDirectiveKind::Update, Start, DirectiveLoc,
                             End) {
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCUpdateConstructClass;
  }
  static OpenACCUpdateConstruct *CreateEmpty(const ASTContext &C,
                                             unsigned NumClauses);
  static OpenACCUpdateConstruct *
  Create(const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
         SourceLocation End, ArrayRef<const OpenACCClause *> Clauses);
};

// This class represents the 'atomic' construct, which has an associated
// statement, but no clauses.
class OpenACCAtomicConstruct final : public OpenACCAssociatedStmtConstruct {

  friend class ASTStmtReader;
  OpenACCAtomicKind AtomicKind = OpenACCAtomicKind::None;

  OpenACCAtomicConstruct(EmptyShell)
      : OpenACCAssociatedStmtConstruct(
            OpenACCAtomicConstructClass, OpenACCDirectiveKind::Atomic,
            SourceLocation{}, SourceLocation{}, SourceLocation{},
            /*AssociatedStmt=*/nullptr) {}

  OpenACCAtomicConstruct(SourceLocation Start, SourceLocation DirectiveLoc,
                         OpenACCAtomicKind AtKind, SourceLocation End,
                         Stmt *AssociatedStmt)
      : OpenACCAssociatedStmtConstruct(OpenACCAtomicConstructClass,
                                       OpenACCDirectiveKind::Atomic, Start,
                                       DirectiveLoc, End, AssociatedStmt),
        AtomicKind(AtKind) {}

  void setAssociatedStmt(Stmt *S) {
    OpenACCAssociatedStmtConstruct::setAssociatedStmt(S);
  }

public:
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpenACCAtomicConstructClass;
  }

  static OpenACCAtomicConstruct *CreateEmpty(const ASTContext &C);
  static OpenACCAtomicConstruct *
  Create(const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
         OpenACCAtomicKind AtKind, SourceLocation End, Stmt *AssociatedStmt);

  OpenACCAtomicKind getAtomicKind() const { return AtomicKind; }
  const Stmt *getAssociatedStmt() const {
    return OpenACCAssociatedStmtConstruct::getAssociatedStmt();
  }
  Stmt *getAssociatedStmt() {
    return OpenACCAssociatedStmtConstruct::getAssociatedStmt();
  }
};

} // namespace clang
#endif // LLVM_CLANG_AST_STMTOPENACC_H
