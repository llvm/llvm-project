//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_LAMBDACAPTURELIFETIMECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_LAMBDACAPTURELIFETIMECHECK_H

#include "../ClangTidyCheck.h"
#include <vector>

namespace clang::tidy::bugprone {

/// Finds lambdas that capture local variables by reference and escape their
/// local scope by being passed to asynchronous sinks or out-of-scope
/// containers.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/lambda-capture-lifetime.html
class LambdaCaptureLifetimeCheck : public ClangTidyCheck {
public:
  LambdaCaptureLifetimeCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }

private:
  const std::vector<StringRef> AsyncClasses;
  const std::vector<StringRef> AsyncFunctions;
  const std::vector<StringRef> StorageClasses;
  const std::vector<StringRef> StorageFunctions;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_LAMBDACAPTURELIFETIMECHECK_H
