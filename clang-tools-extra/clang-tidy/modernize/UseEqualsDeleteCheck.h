//===--- UseEqualsDeleteCheck.h - clang-tidy---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DELETE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DELETE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace modernize {

/// \brief Mark unimplemented private special member functions with '= delete'.
/// \code
///   struct A {
///   private:
///     A(const A&);
///     A& operator=(const A&);
///   };
/// \endcode
/// Is converted to:
/// \code
///   struct A {
///   private:
///     A(const A&) = delete;
///     A& operator=(const A&) = delete;
///   };
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-equals-delete.html
class UseEqualsDeleteCheck : public ClangTidyCheck {
public:
  UseEqualsDeleteCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", 1) != 0) {}
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool IgnoreMacros;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DELETE_H
