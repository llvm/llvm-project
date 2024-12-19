//===--- UseRangesCheck.h - clang-tidy --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USERANGESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USERANGESCHECK_H

#include "../utils/UseRangesCheck.h"

namespace clang::tidy::modernize {

/// Detects calls to standard library iterator algorithms that could be
/// replaced with a ranges version instead
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-ranges.html
class UseRangesCheck : public utils::UseRangesCheck {
public:
  UseRangesCheck(StringRef CheckName, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Options) override;

  ReplacerMap getReplacerMap() const override;

  ArrayRef<std::pair<StringRef, StringRef>>
  getFreeBeginEndMethods() const override;

  std::optional<ReverseIteratorDescriptor>
  getReverseDescriptor() const override;

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;

private:
  bool UseReversePipe;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USERANGESCHECK_H
