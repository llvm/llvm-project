//===- OpenACCClause.h - Classes for OpenACC clauses ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file defines OpenACC AST classes for clauses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_OPENACCCLAUSE_H
#define LLVM_CLANG_AST_OPENACCCLAUSE_H
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtIterator.h"
#include "clang/Basic/OpenACCKinds.h"

namespace clang {
/// This is the base type for all OpenACC Clauses.
class OpenACCClause {
  OpenACCClauseKind Kind;
  SourceRange Location;

protected:
  OpenACCClause(OpenACCClauseKind K, SourceLocation BeginLoc,
                SourceLocation EndLoc)
      : Kind(K), Location(BeginLoc, EndLoc) {}

public:
  OpenACCClauseKind getClauseKind() const { return Kind; }
  SourceLocation getBeginLoc() const { return Location.getBegin(); }
  SourceLocation getEndLoc() const { return Location.getEnd(); }

  static bool classof(const OpenACCClause *) { return true; }

  using child_iterator = StmtIterator;
  using const_child_iterator = ConstStmtIterator;
  using child_range = llvm::iterator_range<child_iterator>;
  using const_child_range = llvm::iterator_range<const_child_iterator>;

  child_range children();
  const_child_range children() const {
    auto Children = const_cast<OpenACCClause *>(this)->children();
    return const_child_range(Children.begin(), Children.end());
  }

  virtual ~OpenACCClause() = default;
};

/// Represents a clause that has a list of parameters.
class OpenACCClauseWithParams : public OpenACCClause {
  /// Location of the '('.
  SourceLocation LParenLoc;

protected:
  OpenACCClauseWithParams(OpenACCClauseKind K, SourceLocation BeginLoc,
                          SourceLocation LParenLoc, SourceLocation EndLoc)
      : OpenACCClause(K, BeginLoc, EndLoc), LParenLoc(LParenLoc) {}

public:
  SourceLocation getLParenLoc() const { return LParenLoc; }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
};

/// A 'default' clause, has the optional 'none' or 'present' argument.
class OpenACCDefaultClause : public OpenACCClauseWithParams {
  friend class ASTReaderStmt;
  friend class ASTWriterStmt;

  OpenACCDefaultClauseKind DefaultClauseKind;

protected:
  OpenACCDefaultClause(OpenACCDefaultClauseKind K, SourceLocation BeginLoc,
                       SourceLocation LParenLoc, SourceLocation EndLoc)
      : OpenACCClauseWithParams(OpenACCClauseKind::Default, BeginLoc, LParenLoc,
                                EndLoc),
        DefaultClauseKind(K) {
    assert((DefaultClauseKind == OpenACCDefaultClauseKind::None ||
            DefaultClauseKind == OpenACCDefaultClauseKind::Present) &&
           "Invalid Clause Kind");
  }

public:
  OpenACCDefaultClauseKind getDefaultClauseKind() const {
    return DefaultClauseKind;
  }

  static OpenACCDefaultClause *Create(const ASTContext &C,
                                      OpenACCDefaultClauseKind K,
                                      SourceLocation BeginLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc);
};

/// Represents one of the handful of classes that has an optional/required
/// 'condition' expression as an argument.
class OpenACCClauseWithCondition : public OpenACCClauseWithParams {
  Expr *ConditionExpr = nullptr;

protected:
  OpenACCClauseWithCondition(OpenACCClauseKind K, SourceLocation BeginLoc,
                             SourceLocation LParenLoc, Expr *ConditionExpr,
                             SourceLocation EndLoc)
      : OpenACCClauseWithParams(K, BeginLoc, LParenLoc, EndLoc),
        ConditionExpr(ConditionExpr) {}

public:
  bool hasConditionExpr() const { return ConditionExpr; }
  const Expr *getConditionExpr() const { return ConditionExpr; }
  Expr *getConditionExpr() { return ConditionExpr; }

  child_range children() {
    if (ConditionExpr)
      return child_range(reinterpret_cast<Stmt **>(&ConditionExpr),
                         reinterpret_cast<Stmt **>(&ConditionExpr + 1));
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    if (ConditionExpr)
      return const_child_range(
          reinterpret_cast<Stmt *const *>(&ConditionExpr),
          reinterpret_cast<Stmt *const *>(&ConditionExpr + 1));
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
};

/// An 'if' clause, which has a required condition expression.
class OpenACCIfClause : public OpenACCClauseWithCondition {
protected:
  OpenACCIfClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                  Expr *ConditionExpr, SourceLocation EndLoc);

public:
  static OpenACCIfClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                 SourceLocation LParenLoc, Expr *ConditionExpr,
                                 SourceLocation EndLoc);
};

/// A 'self' clause, which has an optional condition expression.
class OpenACCSelfClause : public OpenACCClauseWithCondition {
  OpenACCSelfClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                    Expr *ConditionExpr, SourceLocation EndLoc);

public:
  static OpenACCSelfClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                   SourceLocation LParenLoc,
                                   Expr *ConditionExpr, SourceLocation EndLoc);
};

template <class Impl> class OpenACCClauseVisitor {
  Impl &getDerived() { return static_cast<Impl &>(*this); }

public:
  void VisitClauseList(ArrayRef<const OpenACCClause *> List) {
    for (const OpenACCClause *Clause : List)
      Visit(Clause);
  }

  void Visit(const OpenACCClause *C) {
    if (!C)
      return;

    switch (C->getClauseKind()) {
#define VISIT_CLAUSE(CLAUSE_NAME)                                              \
  case OpenACCClauseKind::CLAUSE_NAME:                                         \
    Visit##CLAUSE_NAME##Clause(*cast<OpenACC##CLAUSE_NAME##Clause>(C));        \
    return;
#include "clang/Basic/OpenACCClauses.def"

    default:
      llvm_unreachable("Clause visitor not yet implemented");
    }
    llvm_unreachable("Invalid Clause kind");
  }

#define VISIT_CLAUSE(CLAUSE_NAME)                                              \
  void Visit##CLAUSE_NAME##Clause(                                             \
      const OpenACC##CLAUSE_NAME##Clause &Clause) {                            \
    return getDerived().Visit##CLAUSE_NAME##Clause(Clause);                    \
  }

#include "clang/Basic/OpenACCClauses.def"
};

class OpenACCClausePrinter final
    : public OpenACCClauseVisitor<OpenACCClausePrinter> {
  raw_ostream &OS;

public:
  void VisitClauseList(ArrayRef<const OpenACCClause *> List) {
    for (const OpenACCClause *Clause : List) {
      Visit(Clause);

      if (Clause != List.back())
        OS << ' ';
    }
  }
  OpenACCClausePrinter(raw_ostream &OS) : OS(OS) {}

#define VISIT_CLAUSE(CLAUSE_NAME)                                              \
  void Visit##CLAUSE_NAME##Clause(const OpenACC##CLAUSE_NAME##Clause &Clause);
#include "clang/Basic/OpenACCClauses.def"
};

} // namespace clang

#endif // LLVM_CLANG_AST_OPENACCCLAUSE_H
