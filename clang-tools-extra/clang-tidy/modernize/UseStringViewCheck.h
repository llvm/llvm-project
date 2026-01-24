//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESTRINGVIEWCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESTRINGVIEWCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Looks for functions returning `std::[w|u8|u16|u32]string` and suggests to
/// change it to `std::[...]string_view` if possible and profitable.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/use-string-view.html
class UseStringViewCheck : public ClangTidyCheck {
public:
  UseStringViewCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  StringRef toStringViewTypeStr(StringRef Type) const;
  void parseReplacementStringViewClass(StringRef Options);

  StringRef StringViewClass = "std::string_view";
  StringRef WStringViewClass = "std::wstring_view";
  StringRef U8StringViewClass = "std::u8string_view";
  StringRef U16StringViewClass = "std::u16string_view";
  StringRef U32StringViewClass = "std::u32string_view";

  const std::vector<StringRef> IgnoredFunctions;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESTRINGVIEWCHECK_H
