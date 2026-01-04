//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_RANDOMGENERATORSEEDCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_RANDOMGENERATORSEEDCHECK_H

#include "../ClangTidyCheck.h"
#include <string>

namespace clang::tidy::bugprone {

/// Random number generator must be seeded properly.
///
/// A random number generator initialized with default value or a
/// constant expression is a security vulnerability.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/random-generator-seed.html
class RandomGeneratorSeedCheck : public ClangTidyCheck {
public:
  RandomGeneratorSeedCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  template <class T>
  void checkSeed(const ast_matchers::MatchFinder::MatchResult &Result,
                 const T *Func);

  StringRef RawDisallowedSeedTypes;
  SmallVector<StringRef, 5> DisallowedSeedTypes;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_RANDOMGENERATORSEEDCHECK_H
