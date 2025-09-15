//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_ALIASING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_ALIASING_H

#include "clang/AST/Decl.h"

namespace clang::tidy::utils {

/// Returns whether \p Var has a pointer or reference in \p Func.
///
/// Example:
/// void f() {
///   int n;
///   ...
///   int *p = &n;
/// }
///
/// For `f()` and `n` the function returns ``true`` because `p` is a
/// pointer to `n` created in `f()`.

bool hasPtrOrReferenceInFunc(const Decl *Func, const ValueDecl *Var);

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_ALIASING_H
