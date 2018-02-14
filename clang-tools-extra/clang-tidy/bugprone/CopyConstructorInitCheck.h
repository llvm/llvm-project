//===--- CopyConstructorInitCheck.h - clang-tidy--------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_COPY_CONSTRUCTOR_INIT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_COPY_CONSTRUCTOR_INIT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Finds copy constructors where the ctor don't call the copy constructor of
/// the base class.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-copy-constructor-init.html
class CopyConstructorInitCheck : public ClangTidyCheck {
public:
  CopyConstructorInitCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_COPY_CONSTRUCTOR_INIT_H
