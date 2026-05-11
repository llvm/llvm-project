//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_NARROWINGCONVERSIONS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_NARROWINGCONVERSIONS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"

namespace clang::tidy::utils {

/// Returns \c true if converting \p Init from \p From to \p To is narrowing
bool isNarrowingConversion(QualType From, QualType To, const Expr *Init,
                           const ASTContext &Ctx);

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_NARROWINGCONVERSIONS_H
