//===---------- ASTUtils.cpp - clang-tidy ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ASTUtils.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang {
namespace tidy {
namespace utils {
using namespace ast_matchers;

const FunctionDecl *getSurroundingFunction(ASTContext &Context,
                                           const Stmt &Statement) {
  return selectFirst<const FunctionDecl>(
      "function", match(stmt(hasAncestor(functionDecl().bind("function"))),
                        Statement, Context));
}
} // namespace utils
} // namespace tidy
} // namespace clang
