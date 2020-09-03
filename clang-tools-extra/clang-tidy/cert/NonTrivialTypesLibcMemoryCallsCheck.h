//===--- NonTrivialTypesLibcMemoryCallsCheck.h - clang-tidy -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_NONTRIVIALTYPESLIBCMEMORYCALLSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_NONTRIVIALTYPESLIBCMEMORYCALLSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace cert {

/// Flags use of the `C` standard library functions 'memset', 'memcpy' and
/// 'memcmp' and similar derivatives on non-trivial types.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert-oop57-cpp.html
class NonTrivialTypesLibcMemoryCallsCheck : public ClangTidyCheck {
public:
  NonTrivialTypesLibcMemoryCallsCheck(StringRef Name,
                                      ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus && !LangOpts.ObjC;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const std::string MemSetNames;
  const std::string MemCpyNames;
  const std::string MemCmpNames;
};

} // namespace cert
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_NOTTRIVIALTYPESLIBCMEMORYCALLSCHECK_H
