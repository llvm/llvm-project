//===--- Query.h - clang-query ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_H

#include "QuerySession.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include <string>

namespace clang {
namespace query {

enum OutputKind { OK_Diag, OK_Print, OK_DetailedAST, OK_SrcLoc };

enum QueryKind {
  QK_Invalid,
  QK_NoOp,
  QK_Help,
  QK_Let,
  QK_Match,
  QK_SetBool,
  QK_SetOutputKind,
  QK_SetTraversalKind,
  QK_EnableOutputKind,
  QK_DisableOutputKind,
  QK_Quit
};

class QuerySession;

struct Query : llvm::RefCountedBase<Query> {
  Query(QueryKind Kind) : Kind(Kind) {}
  virtual ~Query();

  /// Perform the query on \p QS and print output to \p OS.
  ///
  /// \return false if an error occurs, otherwise return true.
  virtual bool run(llvm::raw_ostream &OS, QuerySession &QS) const = 0;

  StringRef RemainingContent;
  const QueryKind Kind;
};

typedef llvm::IntrusiveRefCntPtr<Query> QueryRef;

/// Any query which resulted in a parse error.  The error message is in ErrStr.
struct InvalidQuery : Query {
  InvalidQuery(const Twine &ErrStr) : Query(QK_Invalid), ErrStr(ErrStr.str()) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  std::string ErrStr;

  static bool classof(const Query *Q) { return Q->Kind == QK_Invalid; }
};

/// No-op query (i.e. a blank line).
struct NoOpQuery : Query {
  NoOpQuery() : Query(QK_NoOp) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *Q) { return Q->Kind == QK_NoOp; }
};

/// Query for "help".
struct HelpQuery : Query {
  HelpQuery() : Query(QK_Help) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *Q) { return Q->Kind == QK_Help; }
};

/// Query for "quit".
struct QuitQuery : Query {
  QuitQuery() : Query(QK_Quit) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *Q) { return Q->Kind == QK_Quit; }
};

/// Query for "match MATCHER".
struct MatchQuery : Query {
  MatchQuery(StringRef Source,
             const ast_matchers::dynamic::DynTypedMatcher &Matcher)
      : Query(QK_Match), Matcher(Matcher), Source(Source) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  ast_matchers::dynamic::DynTypedMatcher Matcher;

  StringRef Source;

  static bool classof(const Query *Q) { return Q->Kind == QK_Match; }
};

struct LetQuery : Query {
  LetQuery(StringRef Name, const ast_matchers::dynamic::VariantValue &Value)
      : Query(QK_Let), Name(Name), Value(Value) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  std::string Name;
  ast_matchers::dynamic::VariantValue Value;

  static bool classof(const Query *Q) { return Q->Kind == QK_Let; }
};

template <typename T> struct SetQueryKind {};

template <> struct SetQueryKind<bool> {
  static const QueryKind value = QK_SetBool;
};

template <> struct SetQueryKind<OutputKind> {
  static const QueryKind value = QK_SetOutputKind;
};

template <> struct SetQueryKind<TraversalKind> {
  static const QueryKind value = QK_SetTraversalKind;
};

/// Query for "set VAR VALUE".
template <typename T> struct SetQuery : Query {
  SetQuery(T QuerySession::*Var, T Value)
      : Query(SetQueryKind<T>::value), Var(Var), Value(Value) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override {
    QS.*Var = Value;
    return true;
  }

  static bool classof(const Query *Q) {
    return Q->Kind == SetQueryKind<T>::value;
  }

  T QuerySession::*Var;
  T Value;
};

// Implements the exclusive 'set output dump|diag|print' options.
struct SetExclusiveOutputQuery : Query {
  SetExclusiveOutputQuery(bool QuerySession::*Var)
      : Query(QK_SetOutputKind), Var(Var) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override {
    QS.DiagOutput = false;
    QS.DetailedASTOutput = false;
    QS.PrintOutput = false;
    QS.SrcLocOutput = false;
    QS.*Var = true;
    return true;
  }

  static bool classof(const Query *Q) { return Q->Kind == QK_SetOutputKind; }

  bool QuerySession::*Var;
};

// Implements the non-exclusive 'set output dump|diag|print' options.
struct SetNonExclusiveOutputQuery : Query {
  SetNonExclusiveOutputQuery(QueryKind Kind, bool QuerySession::*Var,
                             bool Value)
      : Query(Kind), Var(Var), Value(Value) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override {
    QS.*Var = Value;
    return true;
  }

  bool QuerySession::*Var;
  bool Value;
};

struct EnableOutputQuery : SetNonExclusiveOutputQuery {
  EnableOutputQuery(bool QuerySession::*Var)
      : SetNonExclusiveOutputQuery(QK_EnableOutputKind, Var, true) {}

  static bool classof(const Query *Q) { return Q->Kind == QK_EnableOutputKind; }
};

struct DisableOutputQuery : SetNonExclusiveOutputQuery {
  DisableOutputQuery(bool QuerySession::*Var)
      : SetNonExclusiveOutputQuery(QK_DisableOutputKind, Var, false) {}

  static bool classof(const Query *Q) {
    return Q->Kind == QK_DisableOutputKind;
  }
};

} // namespace query
} // namespace clang

#endif
