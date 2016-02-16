//===--- ProTypeMemberInitCheck.h - clang-tidy-------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_MEMBER_INIT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_MEMBER_INIT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// \brief Checks that builtin or pointer fields are initialized by
/// user-defined constructors.
///
/// Members initialized through function calls in the body of the constructor
/// will result in false positives.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-pro-type-member-init.html
/// TODO: See if 'fixes' for false positives are optimized away by the compiler.
/// TODO: "Issue a diagnostic when constructing an object of a trivially
/// constructible type without () or {} to initialize its members. To fix: Add
/// () or {}."
class ProTypeMemberInitCheck : public ClangTidyCheck {
public:
  ProTypeMemberInitCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_MEMBER_INIT_H
