//=======- ASTUtis.h ---------------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYZER_WEBKIT_ASTUTILS_H
#define LLVM_CLANG_ANALYZER_WEBKIT_ASTUTILS_H

#include "clang/AST/Decl.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"

#include <string>
#include <utility>

namespace clang {
class CXXRecordDecl;
class CXXBaseSpecifier;
class FunctionDecl;
class CXXMethodDecl;
class Expr;

/// If passed expression is of type uncounted pointer/reference we try to find
/// the origin of this pointer. Example: Origin can be a local variable, nullptr
/// constant or this-pointer.
///
/// Certain subexpression nodes represent transformations that don't affect
/// where the memory address originates from. We try to traverse such
/// subexpressions to get to the relevant child nodes. Whenever we encounter a
/// subexpression that either can't be ignored, we don't model its semantics or
/// that has multiple children we stop.
///
/// \p E is an expression of uncounted pointer/reference type.
/// If \p StopAtFirstRefCountedObj is true and we encounter a subexpression that
/// represents ref-counted object during the traversal we return relevant
/// sub-expression and true.
///
/// \returns subexpression that we traversed to and if \p
/// StopAtFirstRefCountedObj is true we also return whether we stopped early.
std::pair<const clang::Expr *, bool>
tryToFindPtrOrigin(const clang::Expr *E, bool StopAtFirstRefCountedObj);

/// For \p E referring to a ref-countable/-counted pointer/reference we return
/// whether it's a safe call argument. Examples: function parameter or
/// this-pointer. The logic relies on the set of recursive rules we enforce for
/// WebKit codebase.
///
/// \returns Whether \p E is a safe call arugment.
bool isASafeCallArg(const clang::Expr *E);

/// \returns name of AST node or empty string.
template <typename T> std::string safeGetName(const T *ASTNode) {
  const auto *const ND = llvm::dyn_cast_or_null<clang::NamedDecl>(ASTNode);
  if (!ND)
    return "";

  // In case F is for example "operator|" the getName() method below would
  // assert.
  if (!ND->getDeclName().isIdentifier())
    return "";

  return ND->getName().str();
}

} // namespace clang

#endif
