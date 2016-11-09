//===--- StringIntegerAssignmentCheck.h - clang-tidy-------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_INTEGER_ASSIGNMENT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_INTEGER_ASSIGNMENT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// Finds instances where an integer is assigned to a string.
///
/// For more details see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-string-assignment.html
class StringIntegerAssignmentCheck : public ClangTidyCheck {
public:
  StringIntegerAssignmentCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_INTEGER_ASSIGNMENT_H
