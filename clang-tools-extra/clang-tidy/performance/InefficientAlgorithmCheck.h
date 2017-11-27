//===--- InefficientAlgorithmCheck.h - clang-tidy----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENTALGORITHMCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENTALGORITHMCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace performance {

/// Warns on inefficient use of STL algorithms on associative containers.
///
/// Associative containers implements some of the algorithms as methods which
/// should be preferred to the algorithms in the algorithm header. The methods
/// can take advanatage of the order of the elements.
class InefficientAlgorithmCheck : public ClangTidyCheck {
public:
  InefficientAlgorithmCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENTALGORITHMCHECK_H
