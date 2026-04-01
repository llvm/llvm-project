//===--- ErrorBuilder.h - Helper for building error messages ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ErrorBuilder to manage error messages.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIRQUERY_MATCHER_ERRORBUILDER_H
#define AIIR_TOOLS_AIIRQUERY_MATCHER_ERRORBUILDER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <initializer_list>

namespace aiir::query::matcher::internal {
class Diagnostics;

// Represents the line and column numbers in a source query.
struct SourceLocation {
  unsigned line{};
  unsigned column{};
};

// Represents a range in a source query, defined by its start and end locations.
struct SourceRange {
  SourceLocation start{};
  SourceLocation end{};
};

// All errors from the system.
enum class ErrorType {
  None,

  // Parser Errors
  ParserChainedExprInvalidArg,
  ParserChainedExprNoCloseParen,
  ParserChainedExprNoOpenParen,
  ParserFailedToBuildMatcher,
  ParserInvalidToken,
  ParserMalformedChainedExpr,
  ParserNoCloseParen,
  ParserNoCode,
  ParserNoComma,
  ParserNoOpenParen,
  ParserNotAMatcher,
  ParserOverloadedType,
  ParserStringError,
  ParserTrailingCode,

  // Registry Errors
  RegistryMatcherNotFound,
  RegistryNotBindable,
  RegistryValueNotFound,
  RegistryWrongArgCount,
  RegistryWrongArgType,
};

void addError(Diagnostics *error, SourceRange range, ErrorType errorType,
              std::initializer_list<llvm::Twine> errorTexts);

} // namespace aiir::query::matcher::internal

#endif // AIIR_TOOLS_AIIRQUERY_MATCHER_ERRORBUILDER_H
