//===--- SlicingCheck.h - clang-tidy-----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_SLICING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_SLICING_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// Flags slicing (incomplete copying of an object's state) of member variables
/// or vtable. See:
///   - https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#es63-dont-slice
///     for the former, and
///   - https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#c145-access-polymorphic-objects-through-pointers-and-references
///     for the latter
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/slicing.html
class SlicingCheck : public ClangTidyCheck {
public:
  SlicingCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void diagnoseSlicedOverriddenMethods(const Expr &Call,
                                       const CXXRecordDecl &DerivedDecl,
                                       const CXXRecordDecl &BaseDecl);
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_SLICING_H
