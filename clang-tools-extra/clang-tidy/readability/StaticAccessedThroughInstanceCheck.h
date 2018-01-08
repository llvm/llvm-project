//===--- StaticAccessedThroughInstanceCheck.h - clang-tidy-------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_STATIC_ACCESSED_THROUGH_INSTANCE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_STATIC_ACCESSED_THROUGH_INSTANCE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// \@brief Checks for member expressions that access static members through
/// instances and replaces them with uses of the appropriate qualified-id.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-static-accessed-through-instance.html
class StaticAccessedThroughInstanceCheck : public ClangTidyCheck {
public:
  StaticAccessedThroughInstanceCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        NameSpecifierNestingThreshold(
            Options.get("NameSpecifierNestingThreshold", 3)) {}

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const unsigned NameSpecifierNestingThreshold;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_STATIC_ACCESSED_THROUGH_INSTANCE_H
