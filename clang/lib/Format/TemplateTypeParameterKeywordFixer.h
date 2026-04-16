//===--- TemplateTypeParameterKeywordFixer.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Rewrites ``typename`` / ``class`` introducing type and template template
/// parameters in ``template <...>`` clauses according to FormatStyle.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_TEMPLATETYPEPARAMETERKEYWORDFIXER_H
#define LLVM_CLANG_LIB_FORMAT_TEMPLATETYPEPARAMETERKEYWORDFIXER_H

#include "TokenAnalyzer.h"

namespace clang {
namespace format {

class TemplateTypeParameterKeywordFixer : public TokenAnalyzer {
public:
  TemplateTypeParameterKeywordFixer(const Environment &Env,
                                    const FormatStyle &Style);

  std::pair<tooling::Replacements, unsigned>
  analyze(TokenAnnotator &Annotator,
          SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
          FormatTokenLexer &Tokens) override;
};

} // namespace format
} // namespace clang

#endif
