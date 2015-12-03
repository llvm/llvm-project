//===--- Matchers.h - clang-tidy-------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"
#include "TypeTraits.h"

namespace clang {
namespace tidy {
namespace matchers {

AST_MATCHER(QualType, isExpensiveToCopy) {
  llvm::Optional<bool> IsExpensive =
      type_traits::isExpensiveToCopy(Node, Finder->getASTContext());
  return IsExpensive && *IsExpensive;
}

} // namespace matchers
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H
