//===--- FloatLoopCounter.h - clang-tidy-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_FLOAT_LOOP_COUNTER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_FLOAT_LOOP_COUNTER_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace cert {

/// This check diagnoses when the loop induction expression of a for loop has
/// floating-point type. The check corresponds to:
/// https://www.securecoding.cert.org/confluence/display/c/FLP30-C.+Do+not+use+floating-point+variables+as+loop+counters
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert-flp30-c.html
class FloatLoopCounter : public ClangTidyCheck {
public:
  FloatLoopCounter(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cert
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_FLOAT_LOOP_COUNTER_H
