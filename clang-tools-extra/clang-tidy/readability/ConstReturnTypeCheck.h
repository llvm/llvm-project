//===--- ConstReturnTypeCheck.h - clang-tidy --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTRETURNTYPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTRETURNTYPECHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// For any function whose return type is const-qualified, suggests removal of
/// the `const` qualifier from that return type.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-const-return-type.html
class ConstReturnTypeCheck : public ClangTidyCheck {
 public:
  using ClangTidyCheck::ClangTidyCheck;
  void registerMatchers(ast_matchers::MatchFinder* finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTRETURNTYPECHECK_H
