//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_RAWMEMORYCALLONNONTRIVIALTYPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_RAWMEMORYCALLONNONTRIVIALTYPECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Flags use of the C standard library functions 'memset', 'memcpy' and
/// 'memcmp' and similar derivatives on non-trivial types.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/raw-memory-call-on-non-trivial-type.html
class RawMemoryCallOnNonTrivialTypeCheck : public ClangTidyCheck {
public:
  RawMemoryCallOnNonTrivialTypeCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus && !LangOpts.ObjC;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const StringRef MemSetNames;
  const StringRef MemCpyNames;
  const StringRef MemCmpNames;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_RAWMEMORYCALLONNONTRIVIALTYPECHECK_H
