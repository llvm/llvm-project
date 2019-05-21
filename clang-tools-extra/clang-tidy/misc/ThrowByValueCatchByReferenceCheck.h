//===--- ThrowByValueCatchByReferenceCheck.h - clang-tidy--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_THROW_BY_VALUE_CATCH_BY_REFERENCE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_THROW_BY_VALUE_CATCH_BY_REFERENCE_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace misc {

///\brief checks for locations that do not throw by value
// or catch by reference.
// The check is C++ only. It checks that all throw locations
// throw by value and not by pointer. Additionally it
// contains an option ("CheckThrowTemporaries", default value "true") that
// checks that thrown objects are anonymous temporaries. It is also
// acceptable for this check to throw string literals.
// This test checks that exceptions are caught by reference
// and not by value or pointer. It will not warn when catching
// pointer to char, wchar_t, char16_t or char32_t. This is
// due to not warning on throwing string literals.
class ThrowByValueCatchByReferenceCheck : public ClangTidyCheck {
public:
  ThrowByValueCatchByReferenceCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void diagnoseThrowLocations(const CXXThrowExpr *throwExpr);
  void diagnoseCatchLocations(const CXXCatchStmt *catchStmt,
                              ASTContext &context);
  bool isFunctionParameter(const DeclRefExpr *declRefExpr);
  bool isCatchVariable(const DeclRefExpr *declRefExpr);
  bool isFunctionOrCatchVar(const DeclRefExpr *declRefExpr);
  const bool CheckAnonymousTemporaries;
  const bool WarnOnLargeObject;
  uint64_t MaxSize; // No `const` because we have to set it in two steps.
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_THROW_BY_VALUE_CATCH_BY_REFERENCE_H
