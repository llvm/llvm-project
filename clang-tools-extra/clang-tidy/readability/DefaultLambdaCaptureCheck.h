//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DEFAULTLAMBDACAPTURECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DEFAULTLAMBDACAPTURECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/** Flags lambdas that use default capture modes
 *
 * For the user-facing documentation see:
 * https://clang.llvm.org/extra/clang-tidy/checks/readability/default-lambda-capture.html
 */
class DefaultLambdaCaptureCheck : public ClangTidyCheck {
public:
  DefaultLambdaCaptureCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DEFAULTLAMBDACAPTURECHECK_H
