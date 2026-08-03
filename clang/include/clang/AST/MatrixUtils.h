//===- MatrixUtils.h - Matrix AST utilities -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines AST-level helper utilities for matrix types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATRIXUTILS_H
#define LLVM_CLANG_AST_MATRIXUTILS_H

#include "clang/AST/Type.h"
#include "clang/Basic/LangOptions.h"

namespace clang {
/// Returns true if matrices of \p T should be laid out in row-major order.
///
/// An explicit layout stored on the matrix type takes precedence over the
/// `-fmatrix-memory-layout=` default carried in \p LangOpts.
inline bool isMatrixRowMajor(const LangOptions &LangOpts, QualType T) {
  if (LangOpts.HLSL && !T.isNull())
    if (const auto *MT = T->getAs<ConstantMatrixType>())
      if (auto Layout = MT->getLayout())
        return *Layout == MatrixType::LayoutKind::RowMajor;
  return LangOpts.getDefaultMatrixMemoryLayout() ==
         LangOptions::MatrixMemoryLayout::MatrixRowMajor;
}
} // namespace clang

#endif // LLVM_CLANG_AST_MATRIXUTILS_H
