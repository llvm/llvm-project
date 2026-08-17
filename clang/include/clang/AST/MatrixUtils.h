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
inline std::optional<HLSLMatrixLayoutType::LayoutKind>
getHLSLMatrixLayout(QualType T) {
  SplitQualType Cur = T.split();
  while (Cur.Ty) {
    if (const auto *LayoutTy = dyn_cast<HLSLMatrixLayoutType>(Cur.Ty))
      return LayoutTy->getLayout();

    if (const auto *ArrayTy = dyn_cast<ArrayType>(Cur.Ty)) {
      Cur = ArrayTy->getElementType().split();
      continue;
    }

    SplitQualType Desugared = Cur.getSingleStepDesugaredType();
    if (Desugared == Cur)
      return std::nullopt;
    Cur = Desugared;
  }
  return std::nullopt;
}

/// Returns true if matrices of \p T should be laid out in row-major order.
///
/// In HLSL mode, explicit layout metadata takes precedence over the
/// `-fmatrix-memory-layout=` default carried in \p LangOpts.
inline bool isMatrixRowMajor(const LangOptions &LangOpts, QualType T) {
  if (LangOpts.HLSL)
    if (auto Layout = getHLSLMatrixLayout(T))
      return *Layout == HLSLMatrixLayoutType::LayoutKind::RowMajor;
  return LangOpts.getDefaultMatrixMemoryLayout() ==
         LangOptions::MatrixMemoryLayout::MatrixRowMajor;
}
} // namespace clang

#endif // LLVM_CLANG_AST_MATRIXUTILS_H
