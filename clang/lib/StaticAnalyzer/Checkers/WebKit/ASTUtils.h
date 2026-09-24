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

#include <functional>
#include <string>
#include <utility>

namespace clang {
class Expr;

/// This function de-facto defines a set of transformations that we consider
/// safe (in heuristical sense). These transformation if passed a safe value as
/// an input should provide a safe value (or an object that provides safe
/// values).
///
/// For more context see Static Analyzer checkers documentation - specifically
/// webkit.UncountedCallArgsChecker checker. Allowed list of transformations:
/// - constructors of ref-counted types (including factory methods)
/// - getters of ref-counted types
/// - member overloaded operators
/// - casts
/// - unary operators like ``&`` or ``*``
///
/// If passed expression is of type uncounted pointer/reference we try to find
/// the "origin" of the pointer value.
/// Origin can be for example a local variable, nullptr, constant or
/// this-pointer.
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
/// Calls \p callback for each origin the traversal reaches, passing the
/// subexpression, whether the traversal recognized it as a safe origin,
/// whether the path to it passed through a temporary that dies at the end of
/// the full-expression (in that case the origin's lifetime guarantee cannot
/// be assumed to extend past the full-expression), and whether the path to it
/// followed at least one [[clang::lifetimebound]] edge. Returns false if any
/// of calls to callbacks returned false. Otherwise true.
///
/// If \p FollowLifetimeBound is true, f(x [[clang::lifetimebound]])
/// traverses into x.
bool tryToFindPtrOrigin(
    const clang::Expr *E, bool StopAtFirstRefCountedObj,
    bool FollowLifetimeBound,
    std::function<bool(const clang::CXXRecordDecl *)> isSafePtr,
    std::function<bool(const clang::QualType)> isSafePtrType,
    std::function<bool(const clang::Decl *)> isSafeGlobalDecl,
    std::function<bool(const clang::Expr *, bool /*IsSafe*/,
                       bool /*OriginDependsOnFullExpressionTemporary*/,
                       bool /*PtrIsLifetimeBoundToOrigin*/)>
        callback);

/// For \p E referring to a ref-countable/-counted pointer/reference we return
/// whether the pointee outlives the current function call. Examples: function
/// parameter or this-pointer. Outliving the call is not by itself sufficient
/// evidence of safety for a model that checks for interior destruction.
///
/// \returns Whether the pointee of \p E outlives the current function call.
bool originOutlivesCall(const clang::Expr *E);

/// \returns true if E is nullptr or __null.
bool isNullPtr(const clang::Expr *E);

/// \returns true if E is a MemberExpr accessing a const smart pointer type.
bool isConstOwnerPtrMemberExpr(const clang::Expr *E);

/// \returns true if E is a MemberExpr accessing a member variable which
/// supports CheckedPtr.
bool isExprToGetCheckedPtrCapableMember(const clang::Expr *E);

/// \returns true if \p E is a [[alloc] init] pattern expression.
/// Sets \p InnerExpr to the inner function call or selector invocation.
bool isAllocInit(const Expr *E, const Expr **InnerExpr = nullptr);

/// \returns ObjCInterfaceDecl from a pointer type.
ObjCInterfaceDecl *getObjCDeclFromObjCPtr(const Type *TypePtr);

/// \returns true if E is a CXXMemberCallExpr which returns a const smart
/// pointer type.
class EnsureFunctionAnalysis {
  using CacheTy = llvm::DenseMap<const FunctionDecl *, bool>;
  mutable CacheTy Cache{};

public:
  bool isACallToEnsureFn(const Expr *E) const;
};

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
