//===--- AvoidNSObjectNewCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_AVOIDNSOBJECTNEWCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_AVOIDNSOBJECTNEWCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace google {
namespace objc {

/// This check finds Objective-C code that uses +new to create object instances,
/// or overrides +new in classes. Both are forbidden by Google's Objective-C
/// style guide.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google-avoid-nsobject-new.html
class AvoidNSObjectNewCheck : public ClangTidyCheck {
public:
  AvoidNSObjectNewCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace objc
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_AVOIDNSOBJECTNEWCHECK_H
