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

/// Represents a clause that has one or more expressions associated with it.
class OpenACCClauseWithExprs : public OpenACCClauseWithParams {
  MutableArrayRef<Expr *> Exprs;

protected:
  OpenACCClauseWithExprs(OpenACCClauseKind K, SourceLocation BeginLoc,
                         SourceLocation LParenLoc, SourceLocation EndLoc)
      : OpenACCClauseWithParams(K, BeginLoc, LParenLoc, EndLoc) {}

  /// Used only for initialization, the leaf class can initialize this to
  /// trailing storage.
  void setExprs(MutableArrayRef<Expr *> NewExprs) {
    assert(Exprs.empty() && "Cannot change Exprs list");
    Exprs = NewExprs;
  }

  /// Gets the entire list of expressions, but leave it to the
  /// individual clauses to expose this how they'd like.
  llvm::ArrayRef<Expr *> getExprs() const { return Exprs; }

public:
  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(Exprs.begin()),
                       reinterpret_cast<Stmt **>(Exprs.end()));
  }

  const_child_range children() const {
    child_range Children =
        const_cast<OpenACCClauseWithExprs *>(this)->children();
    return const_child_range(Children.begin(), Children.end());
  }
};

class OpenACCNumGangsClause final
    : public OpenACCClauseWithExprs,
      public llvm::TrailingObjects<OpenACCNumGangsClause, Expr *> {

  OpenACCNumGangsClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                        ArrayRef<Expr *> IntExprs, SourceLocation EndLoc)
      : OpenACCClauseWithExprs(OpenACCClauseKind::NumGangs, BeginLoc, LParenLoc,
                               EndLoc) {
    std::uninitialized_copy(IntExprs.begin(), IntExprs.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), IntExprs.size()));
  }

public:
  static OpenACCNumGangsClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> IntExprs, SourceLocation EndLoc);

  llvm::ArrayRef<Expr *> getIntExprs() {
    return OpenACCClauseWithExprs::getExprs();
  }

  llvm::ArrayRef<Expr *> getIntExprs() const {
    return OpenACCClauseWithExprs::getExprs();
  }
};

/// Represents one of a handful of clauses that have a single integer
/// expression.
class OpenACCClauseWithSingleIntExpr : public OpenACCClauseWithExprs {
  Expr *IntExpr;

protected:
  OpenACCClauseWithSingleIntExpr(OpenACCClauseKind K, SourceLocation BeginLoc,
                                 SourceLocation LParenLoc, Expr *IntExpr,
                                 SourceLocation EndLoc)
      : OpenACCClauseWithExprs(K, BeginLoc, LParenLoc, EndLoc),
        IntExpr(IntExpr) {
    if (IntExpr)
      setExprs(MutableArrayRef<Expr *>{&this->IntExpr, 1});
  }

public:
  bool hasIntExpr() const { return !getExprs().empty(); }
  const Expr *getIntExpr() const {
    return hasIntExpr() ? getExprs()[0] : nullptr;
  }

  Expr *getIntExpr() { return hasIntExpr() ? getExprs()[0] : nullptr; };
};

class OpenACCNumWorkersClause : public OpenACCClauseWithSingleIntExpr {
  OpenACCNumWorkersClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                          Expr *IntExpr, SourceLocation EndLoc);

public:
  static OpenACCNumWorkersClause *Create(const ASTContext &C,
                                         SourceLocation BeginLoc,
                                         SourceLocation LParenLoc,
                                         Expr *IntExpr, SourceLocation EndLoc);
};

class OpenACCVectorLengthClause : public OpenACCClauseWithSingleIntExpr {
  OpenACCVectorLengthClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                            Expr *IntExpr, SourceLocation EndLoc);

public:
  static OpenACCVectorLengthClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         Expr *IntExpr, SourceLocation EndLoc);
};

class OpenACCAsyncClause : public OpenACCClauseWithSingleIntExpr {
  OpenACCAsyncClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                     Expr *IntExpr, SourceLocation EndLoc);

public:
  static OpenACCAsyncClause *Create(const ASTContext &C,
                                    SourceLocation BeginLoc,
                                    SourceLocation LParenLoc, Expr *IntExpr,
                                    SourceLocation EndLoc);
};

/// Represents a clause with one or more 'var' objects, represented as an expr,
/// as its arguments. Var-list is expected to be stored in trailing storage.
/// For now, we're just storing the original expression in its entirety, unlike
/// OMP which has to do a bunch of work to create a private.
class OpenACCClauseWithVarList : public OpenACCClauseWithExprs {
protected:
  OpenACCClauseWithVarList(OpenACCClauseKind K, SourceLocation BeginLoc,
                           SourceLocation LParenLoc, SourceLocation EndLoc)
      : OpenACCClauseWithExprs(K, BeginLoc, LParenLoc, EndLoc) {}

public:
  ArrayRef<Expr *> getVarList() { return getExprs(); }
  ArrayRef<Expr *> getVarList() const { return getExprs(); }
};

class OpenACCPrivateClause final
    : public OpenACCClauseWithVarList,
      public llvm::TrailingObjects<OpenACCPrivateClause, Expr *> {

  OpenACCPrivateClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                       ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Private, BeginLoc,
                                 LParenLoc, EndLoc) {
    std::uninitialized_copy(VarList.begin(), VarList.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), VarList.size()));
  }

public:
  static OpenACCPrivateClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCFirstPrivateClause final
    : public OpenACCClauseWithVarList,
      public llvm::TrailingObjects<OpenACCFirstPrivateClause, Expr *> {

  OpenACCFirstPrivateClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                            ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::FirstPrivate, BeginLoc,
                                 LParenLoc, EndLoc) {
    std::uninitialized_copy(VarList.begin(), VarList.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), VarList.size()));
  }

public:
  static OpenACCFirstPrivateClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCDevicePtrClause final
    : public OpenACCClauseWithVarList,
      public llvm::TrailingObjects<OpenACCDevicePtrClause, Expr *> {

  OpenACCDevicePtrClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                         ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::DevicePtr, BeginLoc,
                                 LParenLoc, EndLoc) {
    std::uninitialized_copy(VarList.begin(), VarList.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), VarList.size()));
  }

public:
  static OpenACCDevicePtrClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCAttachClause final
    : public OpenACCClauseWithVarList,
      public llvm::TrailingObjects<OpenACCAttachClause, Expr *> {

  OpenACCAttachClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                      ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Attach, BeginLoc, LParenLoc,
                                 EndLoc) {
    std::uninitialized_copy(VarList.begin(), VarList.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), VarList.size()));
  }

public:
  static OpenACCAttachClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCNoCreateClause final
    : public OpenACCClauseWithVarList,
      public llvm::TrailingObjects<OpenACCNoCreateClause, Expr *> {

  OpenACCNoCreateClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                        ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::NoCreate, BeginLoc,
                                 LParenLoc, EndLoc) {
    std::uninitialized_copy(VarList.begin(), VarList.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), VarList.size()));
  }

public:
  static OpenACCNoCreateClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCPresentClause final
    : public OpenACCClauseWithVarList,
      public llvm::TrailingObjects<OpenACCPresentClause, Expr *> {

  OpenACCPresentClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                       ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Present, BeginLoc,
                                 LParenLoc, EndLoc) {
    std::uninitialized_copy(VarList.begin(), VarList.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), VarList.size()));
  }

public:
  static OpenACCPresentClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCCopyClause final
    : public OpenACCClauseWithVarList,
      public llvm::TrailingObjects<OpenACCCopyClause, Expr *> {

  OpenACCCopyClause(OpenACCClauseKind Spelling, SourceLocation BeginLoc,
                    SourceLocation LParenLoc, ArrayRef<Expr *> VarList,
                    SourceLocation EndLoc)
      : OpenACCClauseWithVarList(Spelling, BeginLoc, LParenLoc, EndLoc) {
    assert((Spelling == OpenACCClauseKind::Copy ||
            Spelling == OpenACCClauseKind::PCopy ||
            Spelling == OpenACCClauseKind::PresentOrCopy) &&
           "Invalid clause kind for copy-clause");
    std::uninitialized_copy(VarList.begin(), VarList.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), VarList.size()));
  }

public:
  static OpenACCCopyClause *
  Create(const ASTContext &C, OpenACCClauseKind Spelling,
         SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCCopyInClause final
    : public OpenACCClauseWithVarList,
      public llvm::TrailingObjects<OpenACCCopyInClause, Expr *> {
  bool IsReadOnly;

  OpenACCCopyInClause(OpenACCClauseKind Spelling, SourceLocation BeginLoc,
                      SourceLocation LParenLoc, bool IsReadOnly,
                      ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(Spelling, BeginLoc, LParenLoc, EndLoc),
        IsReadOnly(IsReadOnly) {
    assert((Spelling == OpenACCClauseKind::CopyIn ||
            Spelling == OpenACCClauseKind::PCopyIn ||
            Spelling == OpenACCClauseKind::PresentOrCopyIn) &&
           "Invalid clause kind for copyin-clause");
    std::uninitialized_copy(VarList.begin(), VarList.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), VarList.size()));
  }

public:
  bool isReadOnly() const { return IsReadOnly; }
  static OpenACCCopyInClause *
  Create(const ASTContext &C, OpenACCClauseKind Spelling,
         SourceLocation BeginLoc, SourceLocation LParenLoc, bool IsReadOnly,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCCopyOutClause final
    : public OpenACCClauseWithVarList,
      public llvm::TrailingObjects<OpenACCCopyOutClause, Expr *> {
  bool IsZero;

  OpenACCCopyOutClause(OpenACCClauseKind Spelling, SourceLocation BeginLoc,
                       SourceLocation LParenLoc, bool IsZero,
                       ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(Spelling, BeginLoc, LParenLoc, EndLoc),
        IsZero(IsZero) {
    assert((Spelling == OpenACCClauseKind::CopyOut ||
            Spelling == OpenACCClauseKind::PCopyOut ||
            Spelling == OpenACCClauseKind::PresentOrCopyOut) &&
           "Invalid clause kind for copyout-clause");
    std::uninitialized_copy(VarList.begin(), VarList.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), VarList.size()));
  }

public:
  bool isZero() const { return IsZero; }
  static OpenACCCopyOutClause *
  Create(const ASTContext &C, OpenACCClauseKind Spelling,
         SourceLocation BeginLoc, SourceLocation LParenLoc, bool IsZero,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCCreateClause final
    : public OpenACCClauseWithVarList,
      public llvm::TrailingObjects<OpenACCCreateClause, Expr *> {
  bool IsZero;

  OpenACCCreateClause(OpenACCClauseKind Spelling, SourceLocation BeginLoc,
                      SourceLocation LParenLoc, bool IsZero,
                      ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(Spelling, BeginLoc, LParenLoc, EndLoc),
        IsZero(IsZero) {
    assert((Spelling == OpenACCClauseKind::Create ||
            Spelling == OpenACCClauseKind::PCreate ||
            Spelling == OpenACCClauseKind::PresentOrCreate) &&
           "Invalid clause kind for create-clause");
    std::uninitialized_copy(VarList.begin(), VarList.end(),
                            getTrailingObjects<Expr *>());
    setExprs(MutableArrayRef(getTrailingObjects<Expr *>(), VarList.size()));
  }

public:
  bool isZero() const { return IsZero; }
  static OpenACCCreateClause *
  Create(const ASTContext &C, OpenACCClauseKind Spelling,
         SourceLocation BeginLoc, SourceLocation LParenLoc, bool IsZero,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
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
#define CLAUSE_ALIAS(ALIAS_NAME, CLAUSE_NAME)                                  \
  case OpenACCClauseKind::ALIAS_NAME:                                          \
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
  const PrintingPolicy &Policy;

  void printExpr(const Expr *E);

public:
  void VisitClauseList(ArrayRef<const OpenACCClause *> List) {
    for (const OpenACCClause *Clause : List) {
      Visit(Clause);

      if (Clause != List.back())
        OS << ' ';
    }
  }
  OpenACCClausePrinter(raw_ostream &OS, const PrintingPolicy &Policy)
      : OS(OS), Policy(Policy) {}

#define VISIT_CLAUSE(CLAUSE_NAME)                                              \
  void Visit##CLAUSE_NAME##Clause(const OpenACC##CLAUSE_NAME##Clause &Clause);
#include "clang/Basic/OpenACCClauses.def"
};

} // namespace clang

#endif // LLVM_CLANG_AST_OPENACCCLAUSE_H
