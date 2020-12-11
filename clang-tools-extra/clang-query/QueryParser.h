//===--- QueryParser.h - clang-query ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_PARSER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_PARSER_H

#include "Query.h"
#include "QuerySession.h"
#include "llvm/LineEditor/LineEditor.h"
#include <cstddef>

namespace clang {
namespace query {

class QuerySession;

class QueryParser {
public:
  /// Parse \a Line as a query.
  ///
  /// \return A QueryRef representing the query, which may be an InvalidQuery.
  static QueryRef parse(StringRef Line, const QuerySession &QS);

  /// Compute a list of completions for \a Line assuming a cursor at
  /// \param Pos characters past the start of \a Line, ordered from most
  /// likely to least likely.
  ///
  /// \return A vector of completions for \a Line.
  static std::vector<llvm::LineEditor::Completion>
  complete(StringRef Line, size_t Pos, const QuerySession &QS);

private:
  QueryParser(StringRef Line, const QuerySession &QS)
      : Line(Line), CompletionPos(nullptr), QS(QS) {}

  StringRef lexWord();

  template <typename T> struct LexOrCompleteWord;

  QueryRef parseSetBool(bool QuerySession::*Var);
  QueryRef parseSetTraversalKind(TraversalKind QuerySession::*Var);
  template <typename QueryType> QueryRef parseSetOutputKind();
  QueryRef completeMatcherExpression();

  QueryRef endQuery(QueryRef Q);

  /// Parse [\p Begin,\p End).
  ///
  /// \return A reference to the parsed query object, which may be an
  /// \c InvalidQuery if a parse error occurs.
  QueryRef doParse();

  StringRef Line;

  const char *CompletionPos;
  std::vector<llvm::LineEditor::Completion> Completions;

  const QuerySession &QS;
};

} // namespace query
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_PARSER_H
