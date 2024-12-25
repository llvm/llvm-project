//===-- SmartPointerAccessorCaching.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities to help cache accessors for smart pointer
// like objects.
//
// These should be combined with CachedConstAccessorsLattice.
// Beyond basic const accessors, smart pointers may have the following two
// additional issues:
//
// 1) There may be multiple accessors for the same underlying object, e.g.
//    `operator->`, `operator*`, and `get`. Users may use a mixture of these
//    accessors, so the cache should unify them.
//
// 2) There may be non-const overloads of accessors. They are still safe to
//    cache, as they don't modify the container object.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SMARTPOINTERACCESSORCACHING_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SMARTPOINTERACCESSORCACHING_H

#include <cassert>

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang::dataflow {

/// Matchers:
/// For now, these match on any class with an `operator*` or `operator->`
/// where the return types have a similar shape as std::unique_ptr
/// and std::optional.
///
/// - `*` returns a reference to a type `T`
/// - `->` returns a pointer to `T`
/// - `get` returns a pointer to `T`
/// - `value` returns a reference `T`
///
/// (1) The `T` should all match across the accessors (ignoring qualifiers).
///
/// (2) The specific accessor used in a call isn't required to be const,
///     but the class must have a const overload of each accessor.
///
/// For now, we don't have customization to ignore certain classes.
/// For example, if writing a ClangTidy check for `std::optional`, these
/// would also match `std::optional`. In order to have special handling
/// for `std::optional`, we assume the (Matcher, TransferFunction) case
/// with custom handling is ordered early so that these generic cases
/// do not trigger.
ast_matchers::StatementMatcher isSmartPointerLikeOperatorStar();
ast_matchers::StatementMatcher isSmartPointerLikeOperatorArrow();
ast_matchers::StatementMatcher isSmartPointerLikeValueMethodCall();
ast_matchers::StatementMatcher isSmartPointerLikeGetMethodCall();

} // namespace clang::dataflow

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SMARTPOINTERACCESSORCACHING_H
