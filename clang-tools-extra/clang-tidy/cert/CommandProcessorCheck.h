//===--- CommandInterpreterCheck.h - clang-tidy------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_COMMAND_PROCESSOR_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_COMMAND_PROCESSOR_CHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace cert {

/// Execution of a command processor can lead to security vulnerabilities,
/// and is generally not required. Instead, prefer to launch executables
/// directly via mechanisms that give you more control over what executable is
/// actually launched.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert-env33-c.html
class CommandProcessorCheck : public ClangTidyCheck {
public:
  CommandProcessorCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cert
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_COMMAND_PROCESSOR_CHECK_H
