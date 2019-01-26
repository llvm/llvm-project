//===--- AvoidUnderscoreInGoogletestNameCheck.h - clang-tidy ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_AVOIDUNDERSCOREINGOOGLETESTNAMECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_AVOIDUNDERSCOREINGOOGLETESTNAMECHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {
namespace readability {

// Check for underscores in the names of googletest tests, per
// https://github.com/google/googletest/blob/master/googletest/docs/faq.md#why-should-test-suite-names-and-test-names-not-contain-underscore
class AvoidUnderscoreInGoogletestNameCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;

  void registerPPCallbacks(CompilerInstance &Compiler) override;
};

} // namespace readability
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_AVOIDUNDERSCOREINGOOGLETESTNAMECHECK_H
