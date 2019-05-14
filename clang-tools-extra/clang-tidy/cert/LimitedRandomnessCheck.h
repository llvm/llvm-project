//===--- LimitedRandomnessCheck.h - clang-tidy-------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_LIMITED_RANDOMNESS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_LIMITED_RANDOMNESS_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace cert {

/// Pseudorandom number generators are not genuinely random. The result of the
/// std::rand() function makes no guarantees as to the quality of the random
/// sequence produced.
/// This check warns for the usage of std::rand() function.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert-msc50-cpp.html
class LimitedRandomnessCheck : public ClangTidyCheck {
public:
  LimitedRandomnessCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cert
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_LIMITED_RANDOMNESS_H
