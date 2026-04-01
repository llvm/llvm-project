//===--- QueryParser.h - ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIRQUERY_QUERYPARSER_H
#define AIIR_TOOLS_AIIRQUERY_QUERYPARSER_H

#include "Matcher/Parser.h"
#include "aiir/Query/Query.h"
#include "aiir/Query/QuerySession.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/LineEditor/LineEditor.h"

namespace aiir::query {

class QuerySession;

class QueryParser {
public:
  // Parse line as a query and return a QueryRef representing the query, which
  // may be an InvalidQuery.
  static QueryRef parse(llvm::StringRef line, const QuerySession &qs);

  static std::vector<llvm::LineEditor::Completion>
  complete(llvm::StringRef line, size_t pos, const QuerySession &qs);

private:
  QueryParser(llvm::StringRef line, const QuerySession &qs)
      : line(line), completionPos(nullptr), qs(qs) {}

  llvm::StringRef lexWord();

  template <typename T>
  struct LexOrCompleteWord;

  QueryRef completeMatcherExpression();

  QueryRef endQuery(QueryRef queryRef);

  // Parse [begin, end) and returns a reference to the parsed query object,
  // which may be an InvalidQuery if a parse error occurs.
  QueryRef doParse();

  llvm::StringRef line;

  const char *completionPos;
  std::vector<llvm::LineEditor::Completion> completions;

  const QuerySession &qs;
};

} // namespace aiir::query

#endif // AIIR_TOOLS_AIIRQUERY_QUERYPARSER_H
