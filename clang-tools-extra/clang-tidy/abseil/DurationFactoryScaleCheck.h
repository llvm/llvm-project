//===--- DurationFactoryScaleCheck.h - clang-tidy ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONFACTORYSCALECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONFACTORYSCALECHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace abseil {

/// This check finds cases where the incorrect `Duration` factory function is
/// being used by looking for scaling constants inside the factory argument
/// and suggesting a more appropriate factory.  It also looks for the special
/// case of zero and suggests `ZeroDuration()`.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil-duration-factory-scale.html
class DurationFactoryScaleCheck : public ClangTidyCheck {
public:
  DurationFactoryScaleCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace abseil
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONFACTORYSCALECHECK_H
