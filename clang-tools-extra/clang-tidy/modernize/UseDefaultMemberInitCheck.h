//===--- UseDefaultMemberInitCheck.h - clang-tidy----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_DEFAULT_MEMBER_INIT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_DEFAULT_MEMBER_INIT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Convert a default constructor's member initializers into default member
/// initializers.  Remove member initializers that are the same as a default
/// member initializer.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-default-member-init.html
class UseDefaultMemberInitCheck : public ClangTidyCheck {
public:
  UseDefaultMemberInitCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void checkDefaultInit(const ast_matchers::MatchFinder::MatchResult &Result,
                        const CXXCtorInitializer *Init);
  void checkExistingInit(const ast_matchers::MatchFinder::MatchResult &Result,
                         const CXXCtorInitializer *Init);

  const bool UseAssignment;
  const bool IgnoreMacros;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_DEFAULT_MEMBER_INIT_H
