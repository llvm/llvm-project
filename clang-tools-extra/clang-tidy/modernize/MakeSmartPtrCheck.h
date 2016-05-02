//===--- MakeSmartPtrCheck.h - clang-tidy------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SMART_PTR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SMART_PTR_H

#include "../ClangTidy.h"
#include <string>

namespace clang {
namespace tidy {
namespace modernize {

/// Base class for MakeSharedCheck and MakeUniqueCheck.
class MakeSmartPtrCheck : public ClangTidyCheck {
public:
  MakeSmartPtrCheck(StringRef Name, ClangTidyContext *Context,
                    std::string makeSmartPtrFunctionName);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override final;
  void
  check(const ast_matchers::MatchFinder::MatchResult &Result) override final;

protected:
  using SmartPtrTypeMatcher = ast_matchers::internal::BindableMatcher<QualType>;

  /// Returns matcher that match with different smart pointer types.
  ///
  /// Requires to bind pointer type (qualType) with PointerType string declared
  /// in this class.
  virtual SmartPtrTypeMatcher getSmartPointerTypeMatcher() const = 0;

  static const char PointerType[];
  static const char ConstructorCall[];
  static const char NewExpression[];

private:
  std::string makeSmartPtrFunctionName;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SMART_PTR_H
