//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_HEADERGUARDCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_HEADERGUARDCHECK_H

#include "../utils/HeaderGuard.h"

namespace clang::tidy::misc {

/// Finds and fixes header guards.
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/misc/header-guard.html
class HeaderGuardCheck : public utils::HeaderGuardCheck {
public:
  HeaderGuardCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus || LangOpts.C99;
  }

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;

  bool shouldSuggestEndifComment(StringRef Filename) override;
  bool shouldSuggestToAddHeaderGuard(StringRef Filename) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  std::string getHeaderGuard(StringRef Filename, StringRef OldGuard) override;

  const bool AllowPragmaOnce;
  bool HasPragmaOnce = false;

private:
  const std::vector<StringRef> HeaderDirs;
  const bool EndifComment;
  const StringRef Prefix;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_HEADERGUARDCHECK_H
