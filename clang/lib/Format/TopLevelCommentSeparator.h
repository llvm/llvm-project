//===--- TopLevelCommentSeparator.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares TopLevelCommentSeparator, a TokenAnalyzer that inserts
/// new lines or removes empty lines after the top level comment (i.e. comment
/// block at the top of the source file), usually license text or documentation.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_TOPLEVELCOMMENTSEPARATOR_H
#define LLVM_CLANG_LIB_FORMAT_TOPLEVELCOMMENTSEPARATOR_H

#include "TokenAnalyzer.h"
#include "WhitespaceManager.h"

namespace clang {
namespace format {
class TopLevelCommentSeparator : public TokenAnalyzer {
public:
  TopLevelCommentSeparator(const Environment &Env, const FormatStyle &Style)
      : TokenAnalyzer(Env, Style) {}

  std::pair<tooling::Replacements, unsigned>
  analyze(TokenAnnotator &Annotator,
          SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
          FormatTokenLexer &Tokens) override;

private:
  void separateTopLevelComment(SmallVectorImpl<AnnotatedLine *> &Lines,
                               tooling::Replacements &Result,
                               FormatTokenLexer &Tokens);
};
} // namespace format
} // namespace clang

#endif
