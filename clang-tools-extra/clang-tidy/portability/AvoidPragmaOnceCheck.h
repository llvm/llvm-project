//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPRAGMAONCECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPRAGMAONCECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::portability {

/// Finds uses of ``#pragma once`` and suggests replacing them with standard
/// include guards (``#ifndef``/``#define``/``#endif``) for improved
/// portability.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/portability/avoid-pragma-once.html
class AvoidPragmaOnceCheck : public ClangTidyCheck {
public:
  AvoidPragmaOnceCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus || LangOpts.C99;
  }

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
};

} // namespace clang::tidy::portability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPRAGMAONCECHECK_H
