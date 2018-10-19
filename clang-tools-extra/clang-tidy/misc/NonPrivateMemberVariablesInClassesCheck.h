//===--- NonPrivateMemberVariablesInClassesCheck.h - clang-tidy -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NONPRIVATEMEMBERVARIABLESINCLASSESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NONPRIVATEMEMBERVARIABLESINCLASSESCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// This checker finds classes that not only contain the data
/// (non-static member variables), but also have logic (non-static member
/// functions), and diagnoses all member variables that have any other scope
/// other than `private`. They should be made `private`, and manipulated
/// exclusively via the member functions.
///
/// Optionally, classes with all member variables being `public` could be
/// ignored and optionally all `public` member variables could be ignored.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-non-private-member-variables-in-classes.html
class NonPrivateMemberVariablesInClassesCheck : public ClangTidyCheck {
public:
  NonPrivateMemberVariablesInClassesCheck(StringRef Name,
                                          ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool IgnoreClassesWithAllMemberVariablesBeingPublic;
  const bool IgnorePublicMemberVariables;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NONPRIVATEMEMBERVARIABLESINCLASSESCHECK_H
