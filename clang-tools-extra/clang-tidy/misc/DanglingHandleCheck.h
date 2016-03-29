//===--- DanglingHandleCheck.h - clang-tidy----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_DANGLING_HANDLE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_DANGLING_HANDLE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// Detect dangling references in value handlers like
/// std::experimental::string_view.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-dangling-handle.html
class DanglingHandleCheck : public ClangTidyCheck {
public:
  DanglingHandleCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  void registerMatchersForVariables(ast_matchers::MatchFinder *Finder);
  void registerMatchersForReturn(ast_matchers::MatchFinder *Finder);

  const std::vector<std::string> HandleClasses;
  const ast_matchers::internal::Matcher<RecordDecl> IsAHandle;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_DANGLING_HANDLE_H
