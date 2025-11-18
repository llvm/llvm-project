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
#include "llvm/ADT/STLExtras.h"

#include <utility>
#include <variant>

namespace clang {
/// This is the base type for all OpenACC Clauses.
class OpenACCClause {
  OpenACCClauseKind Kind;
  SourceRange Location;

protected:
  OpenACCClause(OpenACCClauseKind K, SourceLocation BeginLoc,
                SourceLocation EndLoc)
      : Kind(K), Location(BeginLoc, EndLoc) {
    assert(!BeginLoc.isInvalid() && !EndLoc.isInvalid() &&
           "Begin and end location must be valid for OpenACCClause");
      }

public:
  OpenACCClauseKind getClauseKind() const { return Kind; }
  SourceLocation getBeginLoc() const { return Location.getBegin(); }
  SourceLocation getEndLoc() const { return Location.getEnd(); }
  SourceRange getSourceRange() const { return Location; }

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

// Represents the 'auto' clause.
class OpenACCAutoClause : public OpenACCClause {
protected:
  OpenACCAutoClause(SourceLocation BeginLoc, SourceLocation EndLoc)
      : OpenACCClause(OpenACCClauseKind::Auto, BeginLoc, EndLoc) {}

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Auto;
  }

  static OpenACCAutoClause *
  Create(const ASTContext &Ctx, SourceLocation BeginLoc, SourceLocation EndLoc);

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
};

// Represents the 'finalize' clause.
class OpenACCFinalizeClause : public OpenACCClause {
protected:
  OpenACCFinalizeClause(SourceLocation BeginLoc, SourceLocation EndLoc)
      : OpenACCClause(OpenACCClauseKind::Finalize, BeginLoc, EndLoc) {}

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Finalize;
  }

  static OpenACCFinalizeClause *
  Create(const ASTContext &Ctx, SourceLocation BeginLoc, SourceLocation EndLoc);

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
};

// Represents the 'if_present' clause.
class OpenACCIfPresentClause : public OpenACCClause {
protected:
  OpenACCIfPresentClause(SourceLocation BeginLoc, SourceLocation EndLoc)
      : OpenACCClause(OpenACCClauseKind::IfPresent, BeginLoc, EndLoc) {}

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::IfPresent;
  }

  static OpenACCIfPresentClause *
  Create(const ASTContext &Ctx, SourceLocation BeginLoc, SourceLocation EndLoc);

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
};

// Represents the 'independent' clause.
class OpenACCIndependentClause : public OpenACCClause {
protected:
  OpenACCIndependentClause(SourceLocation BeginLoc, SourceLocation EndLoc)
      : OpenACCClause(OpenACCClauseKind::Independent, BeginLoc, EndLoc) {}

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Independent;
  }

  static OpenACCIndependentClause *
  Create(const ASTContext &Ctx, SourceLocation BeginLoc, SourceLocation EndLoc);

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
};
// Represents the 'seq' clause.
class OpenACCSeqClause : public OpenACCClause {
protected:
  OpenACCSeqClause(SourceLocation BeginLoc, SourceLocation EndLoc)
      : OpenACCClause(OpenACCClauseKind::Seq, BeginLoc, EndLoc) {}

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Seq;
  }

  static OpenACCSeqClause *
  Create(const ASTContext &Ctx, SourceLocation BeginLoc, SourceLocation EndLoc);

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
};
// Represents the 'nohost' clause.
class OpenACCNoHostClause : public OpenACCClause {
protected:
  OpenACCNoHostClause(SourceLocation BeginLoc, SourceLocation EndLoc)
      : OpenACCClause(OpenACCClauseKind::NoHost, BeginLoc, EndLoc) {}

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::NoHost;
  }
  static OpenACCNoHostClause *
  Create(const ASTContext &Ctx, SourceLocation BeginLoc, SourceLocation EndLoc);

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
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
  static bool classof(const OpenACCClause *C);

  SourceLocation getLParenLoc() const { return LParenLoc; }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }
};

class OpenACCBindClause final : public OpenACCClauseWithParams {
  std::variant<const StringLiteral *, const IdentifierInfo *> Argument;

  OpenACCBindClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                    const clang::StringLiteral *SL, SourceLocation EndLoc)
      : OpenACCClauseWithParams(OpenACCClauseKind::Bind, BeginLoc, LParenLoc,
                                EndLoc),
        Argument(SL) {}
  OpenACCBindClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                    const IdentifierInfo *ID, SourceLocation EndLoc)
      : OpenACCClauseWithParams(OpenACCClauseKind::Bind, BeginLoc, LParenLoc,
                                EndLoc),
        Argument(ID) {}

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Bind;
  }
  static OpenACCBindClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                   SourceLocation LParenLoc,
                                   const IdentifierInfo *ID,
                                   SourceLocation EndLoc);
  static OpenACCBindClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                   SourceLocation LParenLoc,
                                   const StringLiteral *SL,
                                   SourceLocation EndLoc);

  bool isStringArgument() const {
    return std::holds_alternative<const StringLiteral *>(Argument);
  }

  const StringLiteral *getStringArgument() const {
    return std::get<const StringLiteral *>(Argument);
  }

  bool isIdentifierArgument() const {
    return std::holds_alternative<const IdentifierInfo *>(Argument);
  }

  const IdentifierInfo *getIdentifierArgument() const {
    return std::get<const IdentifierInfo *>(Argument);
  }
};

bool operator==(const OpenACCBindClause &LHS, const OpenACCBindClause &RHS);
inline bool operator!=(const OpenACCBindClause &LHS,
                       const OpenACCBindClause &RHS) {
  return !(LHS == RHS);
}

using DeviceTypeArgument = IdentifierLoc;
/// A 'device_type' or 'dtype' clause, takes a list of either an 'asterisk' or
/// an identifier. The 'asterisk' means 'the rest'.
class OpenACCDeviceTypeClause final
    : public OpenACCClauseWithParams,
      private llvm::TrailingObjects<OpenACCDeviceTypeClause,
                                   DeviceTypeArgument> {
  friend TrailingObjects;
  // Data stored in trailing objects as IdentifierInfo* /SourceLocation pairs. A
  // nullptr IdentifierInfo* represents an asterisk.
  unsigned NumArchs;
  OpenACCDeviceTypeClause(OpenACCClauseKind K, SourceLocation BeginLoc,
                          SourceLocation LParenLoc,
                          ArrayRef<DeviceTypeArgument> Archs,
                          SourceLocation EndLoc)
      : OpenACCClauseWithParams(K, BeginLoc, LParenLoc, EndLoc),
        NumArchs(Archs.size()) {
    assert(
        (K == OpenACCClauseKind::DeviceType || K == OpenACCClauseKind::DType) &&
        "Invalid clause kind for device-type");

    assert(!llvm::any_of(Archs, [](const DeviceTypeArgument &Arg) {
      return Arg.getLoc().isInvalid();
    }) && "Invalid SourceLocation for an argument");

    assert((Archs.size() == 1 ||
            !llvm::any_of(Archs,
                          [](const DeviceTypeArgument &Arg) {
                            return Arg.getIdentifierInfo() == nullptr;
                          })) &&
           "Only a single asterisk version is permitted, and must be the "
           "only one");

    llvm::uninitialized_copy(Archs, getTrailingObjects());
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::DType ||
           C->getClauseKind() == OpenACCClauseKind::DeviceType;
  }
  bool hasAsterisk() const {
    return getArchitectures().size() > 0 &&
           getArchitectures()[0].getIdentifierInfo() == nullptr;
  }

  ArrayRef<DeviceTypeArgument> getArchitectures() const {
    return getTrailingObjects(NumArchs);
  }

  static OpenACCDeviceTypeClause *
  Create(const ASTContext &C, OpenACCClauseKind K, SourceLocation BeginLoc,
         SourceLocation LParenLoc, ArrayRef<DeviceTypeArgument> Archs,
         SourceLocation EndLoc);
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
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Default;
  }
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
  static bool classof(const OpenACCClause *C);

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
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::If;
  }
  static OpenACCIfClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                 SourceLocation LParenLoc, Expr *ConditionExpr,
                                 SourceLocation EndLoc);
};

/// A 'self' clause, which has an optional condition expression, or, in the
/// event of an 'update' directive, contains a 'VarList'.
class OpenACCSelfClause final
    : public OpenACCClauseWithParams,
      private llvm::TrailingObjects<OpenACCSelfClause, Expr *> {
  friend TrailingObjects;
  // Holds whether this HAS a condition expression. Lacks a value if this is NOT
  // a condition-expr self clause.
  std::optional<bool> HasConditionExpr;
  // Holds the number of stored expressions.  In the case of a condition-expr
  // self clause, this is expected to be ONE (and there to be 1 trailing
  // object), whether or not that is null.
  unsigned NumExprs;

  OpenACCSelfClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                    Expr *ConditionExpr, SourceLocation EndLoc);
  OpenACCSelfClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                    ArrayRef<Expr *> VarList, SourceLocation EndLoc);

  // Intentionally internal, meant to be an implementation detail of everything
  // else. All non-internal uses should go through getConditionExpr/getVarList.
  ArrayRef<Expr *> getExprs() const { return getTrailingObjects(NumExprs); }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Self;
  }

  bool isConditionExprClause() const { return HasConditionExpr.has_value(); }
  bool isVarListClause() const { return !isConditionExprClause(); }
  bool isEmptySelfClause() const {
    return (isConditionExprClause() && !hasConditionExpr()) ||
           (!isConditionExprClause() && getVarList().empty());
  }

  bool hasConditionExpr() const {
    assert(HasConditionExpr.has_value() &&
           "VarList Self Clause asked about condition expression");
    return *HasConditionExpr;
  }

  const Expr *getConditionExpr() const {
    assert(HasConditionExpr.has_value() &&
           "VarList Self Clause asked about condition expression");
    assert(getExprs().size() == 1 &&
           "ConditionExpr Self Clause with too many Exprs");
    return getExprs()[0];
  }

  Expr *getConditionExpr() {
    assert(HasConditionExpr.has_value() &&
           "VarList Self Clause asked about condition expression");
    assert(getExprs().size() == 1 &&
           "ConditionExpr Self Clause with too many Exprs");
    return getExprs()[0];
  }

  ArrayRef<Expr *> getVarList() {
    assert(!HasConditionExpr.has_value() &&
           "Condition Expr self clause asked about var list");
    return getExprs();
  }
  ArrayRef<Expr *> getVarList() const {
    assert(!HasConditionExpr.has_value() &&
           "Condition Expr self clause asked about var list");
    return getExprs();
  }

  child_range children() {
    return child_range(
        reinterpret_cast<Stmt **>(getTrailingObjects()),
        reinterpret_cast<Stmt **>(getTrailingObjects() + NumExprs));
  }

  const_child_range children() const {
    child_range Children = const_cast<OpenACCSelfClause *>(this)->children();
    return const_child_range(Children.begin(), Children.end());
  }

  static OpenACCSelfClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                   SourceLocation LParenLoc,
                                   Expr *ConditionExpr, SourceLocation EndLoc);
  static OpenACCSelfClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                   SourceLocation LParenLoc,
                                   ArrayRef<Expr *> ConditionExpr,
                                   SourceLocation EndLoc);
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

  /// Used only for initialization, the leaf class can initialize this to
  /// trailing storage, and initialize the data in the trailing storage as well.
  void setExprs(MutableArrayRef<Expr *> NewStorage, ArrayRef<Expr *> Exprs) {
    assert(NewStorage.size() == Exprs.size());
    llvm::uninitialized_copy(Exprs, NewStorage.begin());
    setExprs(NewStorage);
  }

  /// Gets the entire list of expressions, but leave it to the
  /// individual clauses to expose this how they'd like.
  ArrayRef<Expr *> getExprs() const { return Exprs; }

public:
  static bool classof(const OpenACCClause *C);
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

// Represents the 'devnum' and expressions lists for the 'wait' clause.
class OpenACCWaitClause final
    : public OpenACCClauseWithExprs,
      private llvm::TrailingObjects<OpenACCWaitClause, Expr *> {
  friend TrailingObjects;
  SourceLocation QueuesLoc;
  OpenACCWaitClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                    Expr *DevNumExpr, SourceLocation QueuesLoc,
                    ArrayRef<Expr *> QueueIdExprs, SourceLocation EndLoc)
      : OpenACCClauseWithExprs(OpenACCClauseKind::Wait, BeginLoc, LParenLoc,
                               EndLoc),
        QueuesLoc(QueuesLoc) {
    // The first element of the trailing storage is always the devnum expr,
    // whether it is used or not.
    auto *Exprs = getTrailingObjects();
    llvm::uninitialized_copy(ArrayRef(DevNumExpr), Exprs);
    llvm::uninitialized_copy(QueueIdExprs, Exprs + 1);
    setExprs(getTrailingObjects(QueueIdExprs.size() + 1));
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Wait;
  }
  static OpenACCWaitClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                   SourceLocation LParenLoc, Expr *DevNumExpr,
                                   SourceLocation QueuesLoc,
                                   ArrayRef<Expr *> QueueIdExprs,
                                   SourceLocation EndLoc);

  bool hasQueuesTag() const { return !QueuesLoc.isInvalid(); }
  SourceLocation getQueuesLoc() const { return QueuesLoc; }
  bool hasDevNumExpr() const { return getExprs()[0]; }
  Expr *getDevNumExpr() const { return getExprs()[0]; }
  ArrayRef<Expr *> getQueueIdExprs() {
    return OpenACCClauseWithExprs::getExprs().drop_front();
  }
  ArrayRef<Expr *> getQueueIdExprs() const {
    return OpenACCClauseWithExprs::getExprs().drop_front();
  }
  // If this is a plain `wait` (no parens) this returns 'false'. Else Sema/Parse
  // ensures we have at least one QueueId expression.
  bool hasExprs() const { return getLParenLoc().isValid(); }
};

class OpenACCNumGangsClause final
    : public OpenACCClauseWithExprs,
      private llvm::TrailingObjects<OpenACCNumGangsClause, Expr *> {
  friend TrailingObjects;

  OpenACCNumGangsClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                        ArrayRef<Expr *> IntExprs, SourceLocation EndLoc)
      : OpenACCClauseWithExprs(OpenACCClauseKind::NumGangs, BeginLoc, LParenLoc,
                               EndLoc) {
    setExprs(getTrailingObjects(IntExprs.size()), IntExprs);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::NumGangs;
  }
  static OpenACCNumGangsClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> IntExprs, SourceLocation EndLoc);

  ArrayRef<Expr *> getIntExprs() { return OpenACCClauseWithExprs::getExprs(); }

  ArrayRef<Expr *> getIntExprs() const {
    return OpenACCClauseWithExprs::getExprs();
  }
};

class OpenACCTileClause final
    : public OpenACCClauseWithExprs,
      private llvm::TrailingObjects<OpenACCTileClause, Expr *> {
  friend TrailingObjects;
  OpenACCTileClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                    ArrayRef<Expr *> SizeExprs, SourceLocation EndLoc)
      : OpenACCClauseWithExprs(OpenACCClauseKind::Tile, BeginLoc, LParenLoc,
                               EndLoc) {
    setExprs(getTrailingObjects(SizeExprs.size()), SizeExprs);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Tile;
  }
  static OpenACCTileClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                   SourceLocation LParenLoc,
                                   ArrayRef<Expr *> SizeExprs,
                                   SourceLocation EndLoc);
  ArrayRef<Expr *> getSizeExprs() { return OpenACCClauseWithExprs::getExprs(); }

  ArrayRef<Expr *> getSizeExprs() const {
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
  static bool classof(const OpenACCClause *C);
  bool hasIntExpr() const { return !getExprs().empty(); }
  const Expr *getIntExpr() const {
    return hasIntExpr() ? getExprs()[0] : nullptr;
  }

  Expr *getIntExpr() { return hasIntExpr() ? getExprs()[0] : nullptr; };
};

class OpenACCGangClause final
    : public OpenACCClauseWithExprs,
      private llvm::TrailingObjects<OpenACCGangClause, Expr *, OpenACCGangKind> {
  friend TrailingObjects;
protected:
  OpenACCGangClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                    ArrayRef<OpenACCGangKind> GangKinds,
                    ArrayRef<Expr *> IntExprs, SourceLocation EndLoc);

  OpenACCGangKind getGangKind(unsigned I) const {
    return getTrailingObjects<OpenACCGangKind>()[I];
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Gang;
  }

  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return getNumExprs();
  }

  unsigned getNumExprs() const { return getExprs().size(); }
  std::pair<OpenACCGangKind, const Expr *> getExpr(unsigned I) const {
    return {getGangKind(I), getExprs()[I]};
  }

  bool hasExprOfKind(OpenACCGangKind GK) const {
    for (unsigned I = 0; I < getNumExprs(); ++I) {
      if (getGangKind(I) == GK)
        return true;
    }
    return false;
  }

  static OpenACCGangClause *
  Create(const ASTContext &Ctx, SourceLocation BeginLoc,
         SourceLocation LParenLoc, ArrayRef<OpenACCGangKind> GangKinds,
         ArrayRef<Expr *> IntExprs, SourceLocation EndLoc);
};

class OpenACCWorkerClause : public OpenACCClauseWithSingleIntExpr {
protected:
  OpenACCWorkerClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                      Expr *IntExpr, SourceLocation EndLoc);

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Worker;
  }

  static OpenACCWorkerClause *Create(const ASTContext &Ctx,
                                     SourceLocation BeginLoc,
                                     SourceLocation LParenLoc, Expr *IntExpr,
                                     SourceLocation EndLoc);
};

class OpenACCVectorClause : public OpenACCClauseWithSingleIntExpr {
protected:
  OpenACCVectorClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                      Expr *IntExpr, SourceLocation EndLoc);

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Vector;
  }

  static OpenACCVectorClause *Create(const ASTContext &Ctx,
                                     SourceLocation BeginLoc,
                                     SourceLocation LParenLoc, Expr *IntExpr,
                                     SourceLocation EndLoc);
};

class OpenACCNumWorkersClause : public OpenACCClauseWithSingleIntExpr {
  OpenACCNumWorkersClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                          Expr *IntExpr, SourceLocation EndLoc);

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::NumWorkers;
  }
  static OpenACCNumWorkersClause *Create(const ASTContext &C,
                                         SourceLocation BeginLoc,
                                         SourceLocation LParenLoc,
                                         Expr *IntExpr, SourceLocation EndLoc);
};

class OpenACCVectorLengthClause : public OpenACCClauseWithSingleIntExpr {
  OpenACCVectorLengthClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                            Expr *IntExpr, SourceLocation EndLoc);

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::VectorLength;
  }
  static OpenACCVectorLengthClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         Expr *IntExpr, SourceLocation EndLoc);
};

class OpenACCAsyncClause : public OpenACCClauseWithSingleIntExpr {
  OpenACCAsyncClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                     Expr *IntExpr, SourceLocation EndLoc);

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Async;
  }
  static OpenACCAsyncClause *Create(const ASTContext &C,
                                    SourceLocation BeginLoc,
                                    SourceLocation LParenLoc, Expr *IntExpr,
                                    SourceLocation EndLoc);
};

class OpenACCDeviceNumClause : public OpenACCClauseWithSingleIntExpr {
  OpenACCDeviceNumClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                     Expr *IntExpr, SourceLocation EndLoc);

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::DeviceNum;
  }
  static OpenACCDeviceNumClause *Create(const ASTContext &C,
                                        SourceLocation BeginLoc,
                                        SourceLocation LParenLoc, Expr *IntExpr,
                                        SourceLocation EndLoc);
};

class OpenACCDefaultAsyncClause : public OpenACCClauseWithSingleIntExpr {
  OpenACCDefaultAsyncClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                            Expr *IntExpr, SourceLocation EndLoc);

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::DefaultAsync;
  }
  static OpenACCDefaultAsyncClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         Expr *IntExpr, SourceLocation EndLoc);
};

/// Represents a 'collapse' clause on a 'loop' construct. This clause takes an
/// integer constant expression 'N' that represents how deep to collapse the
/// construct. It also takes an optional 'force' tag that permits intervening
/// code in the loops.
class OpenACCCollapseClause : public OpenACCClauseWithSingleIntExpr {
  bool HasForce = false;

  OpenACCCollapseClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                        bool HasForce, Expr *LoopCount, SourceLocation EndLoc);

public:
  const Expr *getLoopCount() const { return getIntExpr(); }
  Expr *getLoopCount() { return getIntExpr(); }

  bool hasForce() const { return HasForce; }

  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Collapse;
  }

  static OpenACCCollapseClause *Create(const ASTContext &C,
                                       SourceLocation BeginLoc,
                                       SourceLocation LParenLoc, bool HasForce,
                                       Expr *LoopCount, SourceLocation EndLoc);
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
  static bool classof(const OpenACCClause *C);
  ArrayRef<Expr *> getVarList() { return getExprs(); }
  ArrayRef<Expr *> getVarList() const { return getExprs(); }
};

// Represents all the data needed for recipe generation.  The declaration and
// init are stored separately, because in the case of subscripts, we do the
// alloca at the level of the base, and the init at the element level.
struct OpenACCPrivateRecipe {
  VarDecl *AllocaDecl;

  OpenACCPrivateRecipe(VarDecl *A) : AllocaDecl(A) {}

  bool isSet() const { return AllocaDecl; }

  static OpenACCPrivateRecipe Empty() {
    return OpenACCPrivateRecipe(/*AllocaDecl=*/nullptr);
  }
};

class OpenACCPrivateClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCPrivateClause, Expr *,
                                    OpenACCPrivateRecipe> {
  friend TrailingObjects;

  OpenACCPrivateClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                       ArrayRef<Expr *> VarList,
                       ArrayRef<OpenACCPrivateRecipe> InitRecipes,
                       SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Private, BeginLoc,
                                 LParenLoc, EndLoc) {
    assert(VarList.size() == InitRecipes.size());
    setExprs(getTrailingObjects<Expr *>(VarList.size()), VarList);
    llvm::uninitialized_copy(InitRecipes,
                             getTrailingObjects<OpenACCPrivateRecipe>());
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Private;
  }
  // Gets a list of 'made up' `VarDecl` objects that can be used by codegen to
  // ensure that we properly initialize each of these variables.
  ArrayRef<OpenACCPrivateRecipe> getInitRecipes() {
    return ArrayRef<OpenACCPrivateRecipe>{
        getTrailingObjects<OpenACCPrivateRecipe>(), getExprs().size()};
  }

  ArrayRef<OpenACCPrivateRecipe> getInitRecipes() const {
    return ArrayRef<OpenACCPrivateRecipe>{
        getTrailingObjects<OpenACCPrivateRecipe>(), getExprs().size()};
  }

  static OpenACCPrivateClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, ArrayRef<OpenACCPrivateRecipe> InitRecipes,
         SourceLocation EndLoc);

  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return getExprs().size();
  }
};

// A 'pair' to stand in for the recipe.  RecipeDecl is the main declaration, and
// InitFromTemporary is the 'temp' declaration we put in to be 'copied from'.
struct OpenACCFirstPrivateRecipe {
  VarDecl *AllocaDecl;
  VarDecl *InitFromTemporary;
  OpenACCFirstPrivateRecipe(VarDecl *A, VarDecl *T)
      : AllocaDecl(A), InitFromTemporary(T) {
    assert(!InitFromTemporary || InitFromTemporary->getInit() == nullptr);
  }

  bool isSet() const { return AllocaDecl; }

  static OpenACCFirstPrivateRecipe Empty() {
    return OpenACCFirstPrivateRecipe(/*AllocaDecl=*/nullptr,
                                     /*InitFromTemporary=*/nullptr);
  }
};

class OpenACCFirstPrivateClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCFirstPrivateClause, Expr *,
                                    OpenACCFirstPrivateRecipe> {
  friend TrailingObjects;

  OpenACCFirstPrivateClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                            ArrayRef<Expr *> VarList,
                            ArrayRef<OpenACCFirstPrivateRecipe> InitRecipes,
                            SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::FirstPrivate, BeginLoc,
                                 LParenLoc, EndLoc) {
    assert(VarList.size() == InitRecipes.size());
    setExprs(getTrailingObjects<Expr *>(VarList.size()), VarList);
    llvm::uninitialized_copy(InitRecipes,
                             getTrailingObjects<OpenACCFirstPrivateRecipe>());
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::FirstPrivate;
  }

  // Gets a list of 'made up' `VarDecl` objects that can be used by codegen to
  // ensure that we properly initialize each of these variables.
  ArrayRef<OpenACCFirstPrivateRecipe> getInitRecipes() {
    return ArrayRef<OpenACCFirstPrivateRecipe>{
        getTrailingObjects<OpenACCFirstPrivateRecipe>(), getExprs().size()};
  }

  ArrayRef<OpenACCFirstPrivateRecipe> getInitRecipes() const {
    return ArrayRef<OpenACCFirstPrivateRecipe>{
        getTrailingObjects<OpenACCFirstPrivateRecipe>(), getExprs().size()};
  }

  static OpenACCFirstPrivateClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList,
         ArrayRef<OpenACCFirstPrivateRecipe> InitRecipes,
         SourceLocation EndLoc);

  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return getExprs().size();
  }
};

class OpenACCDevicePtrClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCDevicePtrClause, Expr *> {
  friend TrailingObjects;

  OpenACCDevicePtrClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                         ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::DevicePtr, BeginLoc,
                                 LParenLoc, EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::DevicePtr;
  }
  static OpenACCDevicePtrClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCAttachClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCAttachClause, Expr *> {
  friend TrailingObjects;

  OpenACCAttachClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                      ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Attach, BeginLoc, LParenLoc,
                                 EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Attach;
  }
  static OpenACCAttachClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCDetachClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCDetachClause, Expr *> {
  friend TrailingObjects;

  OpenACCDetachClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                      ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Detach, BeginLoc, LParenLoc,
                                 EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Detach;
  }
  static OpenACCDetachClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCDeleteClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCDeleteClause, Expr *> {
  friend TrailingObjects;

  OpenACCDeleteClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                      ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Delete, BeginLoc, LParenLoc,
                                 EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Delete;
  }
  static OpenACCDeleteClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCUseDeviceClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCUseDeviceClause, Expr *> {
  friend TrailingObjects;

  OpenACCUseDeviceClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                         ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::UseDevice, BeginLoc,
                                 LParenLoc, EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::UseDevice;
  }
  static OpenACCUseDeviceClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCNoCreateClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCNoCreateClause, Expr *> {
  friend TrailingObjects;

  OpenACCNoCreateClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                        ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::NoCreate, BeginLoc,
                                 LParenLoc, EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::NoCreate;
  }
  static OpenACCNoCreateClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCPresentClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCPresentClause, Expr *> {
  friend TrailingObjects;

  OpenACCPresentClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                       ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Present, BeginLoc,
                                 LParenLoc, EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Present;
  }
  static OpenACCPresentClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};
class OpenACCHostClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCHostClause, Expr *> {
  friend TrailingObjects;

  OpenACCHostClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                    ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Host, BeginLoc, LParenLoc,
                                 EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Host;
  }
  static OpenACCHostClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                   SourceLocation LParenLoc,
                                   ArrayRef<Expr *> VarList,
                                   SourceLocation EndLoc);
};

class OpenACCDeviceClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCDeviceClause, Expr *> {
  friend TrailingObjects;

  OpenACCDeviceClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                      ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Device, BeginLoc, LParenLoc,
                                 EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Device;
  }
  static OpenACCDeviceClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         ArrayRef<Expr *> VarList, SourceLocation EndLoc);
};

class OpenACCCopyClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCCopyClause, Expr *> {
  friend TrailingObjects;
  OpenACCModifierKind Modifiers;

  OpenACCCopyClause(OpenACCClauseKind Spelling, SourceLocation BeginLoc,
                    SourceLocation LParenLoc, OpenACCModifierKind Mods,
                    ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(Spelling, BeginLoc, LParenLoc, EndLoc),
        Modifiers(Mods) {
    assert((Spelling == OpenACCClauseKind::Copy ||
            Spelling == OpenACCClauseKind::PCopy ||
            Spelling == OpenACCClauseKind::PresentOrCopy) &&
           "Invalid clause kind for copy-clause");
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Copy ||
           C->getClauseKind() == OpenACCClauseKind::PCopy ||
           C->getClauseKind() == OpenACCClauseKind::PresentOrCopy;
  }
  static OpenACCCopyClause *
  Create(const ASTContext &C, OpenACCClauseKind Spelling,
         SourceLocation BeginLoc, SourceLocation LParenLoc,
         OpenACCModifierKind Mods, ArrayRef<Expr *> VarList,
         SourceLocation EndLoc);

  OpenACCModifierKind getModifierList() const { return Modifiers; }
};

class OpenACCCopyInClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCCopyInClause, Expr *> {
  friend TrailingObjects;
  OpenACCModifierKind Modifiers;

  OpenACCCopyInClause(OpenACCClauseKind Spelling, SourceLocation BeginLoc,
                      SourceLocation LParenLoc, OpenACCModifierKind Mods,
                      ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(Spelling, BeginLoc, LParenLoc, EndLoc),
        Modifiers(Mods) {
    assert((Spelling == OpenACCClauseKind::CopyIn ||
            Spelling == OpenACCClauseKind::PCopyIn ||
            Spelling == OpenACCClauseKind::PresentOrCopyIn) &&
           "Invalid clause kind for copyin-clause");
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::CopyIn ||
           C->getClauseKind() == OpenACCClauseKind::PCopyIn ||
           C->getClauseKind() == OpenACCClauseKind::PresentOrCopyIn;
  }
  OpenACCModifierKind getModifierList() const { return Modifiers; }
  static OpenACCCopyInClause *
  Create(const ASTContext &C, OpenACCClauseKind Spelling,
         SourceLocation BeginLoc, SourceLocation LParenLoc,
         OpenACCModifierKind Mods, ArrayRef<Expr *> VarList,
         SourceLocation EndLoc);
};

class OpenACCCopyOutClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCCopyOutClause, Expr *> {
  friend TrailingObjects;
  OpenACCModifierKind Modifiers;

  OpenACCCopyOutClause(OpenACCClauseKind Spelling, SourceLocation BeginLoc,
                       SourceLocation LParenLoc, OpenACCModifierKind Mods,
                       ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(Spelling, BeginLoc, LParenLoc, EndLoc),
        Modifiers(Mods) {
    assert((Spelling == OpenACCClauseKind::CopyOut ||
            Spelling == OpenACCClauseKind::PCopyOut ||
            Spelling == OpenACCClauseKind::PresentOrCopyOut) &&
           "Invalid clause kind for copyout-clause");
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::CopyOut ||
           C->getClauseKind() == OpenACCClauseKind::PCopyOut ||
           C->getClauseKind() == OpenACCClauseKind::PresentOrCopyOut;
  }
  OpenACCModifierKind getModifierList() const { return Modifiers; }
  static OpenACCCopyOutClause *
  Create(const ASTContext &C, OpenACCClauseKind Spelling,
         SourceLocation BeginLoc, SourceLocation LParenLoc,
         OpenACCModifierKind Mods, ArrayRef<Expr *> VarList,
         SourceLocation EndLoc);
};

class OpenACCCreateClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCCreateClause, Expr *> {
  friend TrailingObjects;
  OpenACCModifierKind Modifiers;

  OpenACCCreateClause(OpenACCClauseKind Spelling, SourceLocation BeginLoc,
                      SourceLocation LParenLoc, OpenACCModifierKind Mods,
                      ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(Spelling, BeginLoc, LParenLoc, EndLoc),
        Modifiers(Mods) {
    assert((Spelling == OpenACCClauseKind::Create ||
            Spelling == OpenACCClauseKind::PCreate ||
            Spelling == OpenACCClauseKind::PresentOrCreate) &&
           "Invalid clause kind for create-clause");
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Create ||
           C->getClauseKind() == OpenACCClauseKind::PCreate ||
           C->getClauseKind() == OpenACCClauseKind::PresentOrCreate;
  }
  OpenACCModifierKind getModifierList() const { return Modifiers; }
  static OpenACCCreateClause *
  Create(const ASTContext &C, OpenACCClauseKind Spelling,
         SourceLocation BeginLoc, SourceLocation LParenLoc,
         OpenACCModifierKind Mods, ArrayRef<Expr *> VarList,
         SourceLocation EndLoc);
};

// A structure to stand in for the recipe on a reduction.  RecipeDecl is the
// 'main' declaration used for initializaiton, which is fixed.
struct OpenACCReductionRecipe {
  VarDecl *AllocaDecl;

  // A combiner recipe is represented by an operation expression.  However, in
  // order to generate these properly, we have to make up a LHS and a RHS
  // expression for the purposes of generation.
  struct CombinerRecipe {
    VarDecl *LHS;
    VarDecl *RHS;
    Expr *Op;
  };

  // Contains a collection of the recipe elements we need for the combiner:
  // -For Scalars, there will be 1 element, just the combiner for that scalar.
  // -For a struct with a valid operator, this will be 1 element, just that
  //  call.
  // -For a struct without the operator, this will be 1 element per field, which
  //  should be the combiner for that element.
  // -For an array of any of the above, it will be the above for the element.
  // Note: These are necessarily stored in either Trailing Storage (when in the
  // AST), or in a separate collection when being semantically analyzed.
  llvm::ArrayRef<CombinerRecipe> CombinerRecipes;

  bool isSet() const { return AllocaDecl; }

private:
  friend class OpenACCReductionClause;
  OpenACCReductionRecipe(VarDecl *A, llvm::ArrayRef<CombinerRecipe> Combiners)
      : AllocaDecl(A), CombinerRecipes(Combiners) {}
};

// A version of the above that is used for semantic analysis, at a time before
// the OpenACCReductionClause node has been created.  This one has storage for
// the CombinerRecipe, since Trailing storage for it doesn't exist yet.
struct OpenACCReductionRecipeWithStorage {
  VarDecl *AllocaDecl;
  llvm::SmallVector<OpenACCReductionRecipe::CombinerRecipe, 1> CombinerRecipes;

  OpenACCReductionRecipeWithStorage(
      VarDecl *A,
      llvm::ArrayRef<OpenACCReductionRecipe::CombinerRecipe> Combiners)
      : AllocaDecl(A), CombinerRecipes(Combiners) {}

  static OpenACCReductionRecipeWithStorage Empty() {
    return OpenACCReductionRecipeWithStorage(/*AllocaDecl=*/nullptr, {});
  }
};

class OpenACCReductionClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCReductionClause, Expr *,
                                    OpenACCReductionRecipe,
                                    OpenACCReductionRecipe::CombinerRecipe> {
  friend TrailingObjects;
  OpenACCReductionOperator Op;

  OpenACCReductionClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                         OpenACCReductionOperator Operator,
                         ArrayRef<Expr *> VarList,
                         ArrayRef<OpenACCReductionRecipeWithStorage> Recipes,
                         SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Reduction, BeginLoc,
                                 LParenLoc, EndLoc),
        Op(Operator) {
    assert(VarList.size() == Recipes.size());
    setExprs(getTrailingObjects<Expr *>(VarList.size()), VarList);

    // Since we're using trailing storage on this node to store the 'combiner'
    // recipes of the Reduction Recipes (which have a 1:M relationship), we need
    // to ensure we get the ArrayRef of each of our combiner 'correct'.
    OpenACCReductionRecipe::CombinerRecipe *CurCombinerLoc =
        getTrailingObjects<OpenACCReductionRecipe::CombinerRecipe>();
    for (const auto &[Idx, R] : llvm::enumerate(Recipes)) {

      // ArrayRef to the 'correct' data location in trailing storage.
      llvm::MutableArrayRef<OpenACCReductionRecipe::CombinerRecipe>
          NewCombiners{CurCombinerLoc, R.CombinerRecipes.size()};
      CurCombinerLoc += R.CombinerRecipes.size();

      llvm::uninitialized_copy(R.CombinerRecipes, NewCombiners.begin());

      // Placement new into the correct location in trailng storage.
      new (&getTrailingObjects<OpenACCReductionRecipe>()[Idx])
          OpenACCReductionRecipe(R.AllocaDecl, NewCombiners);
    }
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Reduction;
  }

  ArrayRef<OpenACCReductionRecipe> getRecipes() {
    return ArrayRef<OpenACCReductionRecipe>{
        getTrailingObjects<OpenACCReductionRecipe>(), getExprs().size()};
  }

  ArrayRef<OpenACCReductionRecipe> getRecipes() const {
    return ArrayRef<OpenACCReductionRecipe>{
        getTrailingObjects<OpenACCReductionRecipe>(), getExprs().size()};
  }

  static OpenACCReductionClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
         OpenACCReductionOperator Operator, ArrayRef<Expr *> VarList,
         ArrayRef<OpenACCReductionRecipeWithStorage> Recipes,
         SourceLocation EndLoc);

  OpenACCReductionOperator getReductionOp() const { return Op; }

  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return getExprs().size();
  }
  size_t numTrailingObjects(OverloadToken<OpenACCReductionRecipe>) const {
    return getExprs().size();
  }
};

class OpenACCLinkClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCLinkClause, Expr *> {
  friend TrailingObjects;

  OpenACCLinkClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                    ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::Link, BeginLoc, LParenLoc,
                                 EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::Link;
  }

  static OpenACCLinkClause *Create(const ASTContext &C, SourceLocation BeginLoc,
                                   SourceLocation LParenLoc,
                                   ArrayRef<Expr *> VarList,
                                   SourceLocation EndLoc);
};

class OpenACCDeviceResidentClause final
    : public OpenACCClauseWithVarList,
      private llvm::TrailingObjects<OpenACCDeviceResidentClause, Expr *> {
  friend TrailingObjects;

  OpenACCDeviceResidentClause(SourceLocation BeginLoc, SourceLocation LParenLoc,
                              ArrayRef<Expr *> VarList, SourceLocation EndLoc)
      : OpenACCClauseWithVarList(OpenACCClauseKind::DeviceResident, BeginLoc,
                                 LParenLoc, EndLoc) {
    setExprs(getTrailingObjects(VarList.size()), VarList);
  }

public:
  static bool classof(const OpenACCClause *C) {
    return C->getClauseKind() == OpenACCClauseKind::DeviceResident;
  }

  static OpenACCDeviceResidentClause *
  Create(const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
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
    getDerived().Visit##CLAUSE_NAME##Clause(                                   \
        *cast<OpenACC##CLAUSE_NAME##Clause>(C));                               \
    return;
#define CLAUSE_ALIAS(ALIAS_NAME, CLAUSE_NAME, DEPRECATED)                      \
  case OpenACCClauseKind::ALIAS_NAME:                                          \
    getDerived().Visit##CLAUSE_NAME##Clause(                                   \
        *cast<OpenACC##CLAUSE_NAME##Clause>(C));                               \
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
    return getDerived().VisitClause(Clause);                                   \
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
