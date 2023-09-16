//===--- UseStdNumbersCheck.h - clang-tidy ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESTDNUMBERSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESTDNUMBERSCHECK_H

#include "../utils/TransformerClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Finds constants and function calls to math functions that can be replaced
/// with c++20's mathematical constants ('numbers' header). Does not match the
/// use of variables or macros with that value and instead offers a replacement
/// at the definition of said variables and macros.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-std-numbers.html
class UseStdNumbersCheck : public utils::TransformerClangTidyCheck {
public:
  UseStdNumbersCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus20;
  }

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESTDNUMBERSCHECK_H
