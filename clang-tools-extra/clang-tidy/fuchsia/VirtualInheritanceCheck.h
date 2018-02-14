//===--- VirtualInheritanceCheck.h - clang-tidy------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_VIRTUAL_INHERITANCE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_VIRTUAL_INHERITANCE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace fuchsia {

/// Defining classes with virtual inheritance is disallowed.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-virtual-inheritance.html
class VirtualInheritanceCheck : public ClangTidyCheck {
 public:
  VirtualInheritanceCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

}  // namespace fuchsia
}  // namespace tidy
}  // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_VIRTUAL_INHERITANCE_H
