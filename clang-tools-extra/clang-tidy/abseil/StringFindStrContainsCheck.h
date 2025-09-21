//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_STRINGFINDSTRCONTAINSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_STRINGFINDSTRCONTAINSCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/TransformerClangTidyCheck.h"

namespace clang::tidy::abseil {

/// Finds s.find(...) == string::npos comparisons (for various string-like
/// types) and suggests replacing with absl::StrContains.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil/string-find-str-contains.html
class StringFindStrContainsCheck : public utils::TransformerClangTidyCheck {
public:
  StringFindStrContainsCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const std::vector<StringRef> StringLikeClassesOption;
  const StringRef AbseilStringsMatchHeaderOption;
};

} // namespace clang::tidy::abseil

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_STRINGFINDSTRCONTAINSCHECK_H
