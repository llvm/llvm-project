//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USECONCISEPREPROCESSORDIRECTIVESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USECONCISEPREPROCESSORDIRECTIVESCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Finds uses of ``#if`` that can be simplified to ``#ifdef`` or ``#ifndef``
/// and, since C23 and C++23, uses of ``#elif`` that can be simplified to
/// ``#elifdef`` or ``#elifndef``.
///
/// User-facing documentation:
/// https://clang.llvm.org/extra/clang-tidy/checks/readability/use-concise-preprocessor-directives.html
class UseConcisePreprocessorDirectivesCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return true;
  }
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USECONCISEPREPROCESSORDIRECTIVESCHECK_H
