//===--- IncludesSeparator.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares IncludesSeparator, a TokenAnalyzer that inserts
/// new lines or removes empty lines after an includes area.
/// An includes area is a list of consecutive include statements.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_INCLUDESSEPARATOR_H
#define LLVM_CLANG_LIB_FORMAT_INCLUDESSEPARATOR_H

#include "TokenAnalyzer.h"
#include "WhitespaceManager.h"

namespace clang {
namespace format {
class IncludesSeparator : public TokenAnalyzer {
public:
  IncludesSeparator(const Environment &Env, const FormatStyle &Style)
      : TokenAnalyzer(Env, Style) {}

  std::pair<tooling::Replacements, unsigned>
  analyze(TokenAnnotator &Annotator,
          SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
          FormatTokenLexer &Tokens) override;

private:
  void separateIncludes(SmallVectorImpl<AnnotatedLine *> &Lines,
                        tooling::Replacements &Result,
                        FormatTokenLexer &Tokens);
};
} // namespace format
} // namespace clang

#endif
