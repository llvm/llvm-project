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

  /// The list of clauses.  This is stored here as an ArrayRef, as this is the
  /// most convienient place to access the list, however the list itself should
  /// be stored in leaf nodes, likely in trailing-storage.
  MutableArrayRef<const OpenACCClause *> Clauses;

protected:
  OpenACCConstructStmt(StmtClass SC, OpenACCDirectiveKind K,
                       SourceLocation Start, SourceLocation End)
      : Stmt(SC), Kind(K), Range(Start, End) {}

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
                                 SourceLocation Start, SourceLocation End,
                                 Stmt *AssocStmt)
      : OpenACCConstructStmt(SC, K, Start, End), AssociatedStmt(AssocStmt) {}

  void setAssociatedStmt(Stmt *S) { AssociatedStmt = S; }
  Stmt *getAssociatedStmt() { return AssociatedStmt; }
  const Stmt *getAssociatedStmt() const {
    return const_cast<OpenACCAssociatedStmtConstruct *>(this)
        ->getAssociatedStmt();
  }

public:
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
      public llvm::TrailingObjects<OpenACCComputeConstruct,
                                   const OpenACCClause *> {
  friend class ASTStmtWriter;
  friend class ASTStmtReader;
  friend class ASTContext;
  OpenACCComputeConstruct(unsigned NumClauses)
      : OpenACCAssociatedStmtConstruct(OpenACCComputeConstructClass,
                                       OpenACCDirectiveKind::Invalid,
                                       SourceLocation{}, SourceLocation{},
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
                          SourceLocation End,
                          ArrayRef<const OpenACCClause *> Clauses,
                          Stmt *StructuredBlock)
      : OpenACCAssociatedStmtConstruct(OpenACCComputeConstructClass, K, Start,
                                       End, StructuredBlock) {
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
         SourceLocation EndLoc, ArrayRef<const OpenACCClause *> Clauses,
         Stmt *StructuredBlock);

  Stmt *getStructuredBlock() { return getAssociatedStmt(); }
  const Stmt *getStructuredBlock() const {
    return const_cast<OpenACCComputeConstruct *>(this)->getStructuredBlock();
  }
};
} // namespace clang
#endif // LLVM_CLANG_AST_STMTOPENACC_H
