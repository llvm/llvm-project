//===--- MakeFunctionToDirectCheck.h - clang-tidy --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKEFUNCTIONTODIRECTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKEFUNCTIONTODIRECTCHECK_H

#include "../utils/TransformerClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Converts std::make_* function calls to direct constructor calls using
/// class template argument deduction (CTAD).
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/make-direct.html
class MakeFunctionToDirectCheck : public utils::TransformerClangTidyCheck {
public:
  MakeFunctionToDirectCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const bool CheckMakePair;
  const bool CheckMakeOptional;
  const bool CheckMakeTuple;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKEFUNCTIONTODIRECTCHECK_H