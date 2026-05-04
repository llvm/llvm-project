//===- MatrixUtils.h - Matrix AST utilities ---------------------*- C++ -*-===//
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
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/LangOptions.h"

namespace clang {
/// Returns true if matrices of \p T should be laid out in row-major order.
///
/// In HLSL mode, an `HLSLRowMajor` / `HLSLColumnMajor` AttributedType anywhere
/// in the sugar chain of \p T (imprinted by Sema when a source decl carries
/// `[[hlsl::row_major]]` / `[[hlsl::column_major]]`) takes precedence over the
/// `-fmatrix-memory-layout=` default carried in \p LangOpts. Otherwise the
/// LangOptions default is used.
inline bool isMatrixRowMajor(const LangOptions &LangOpts, QualType T) {
  if (LangOpts.HLSL && !T.isNull()) {
    QualType Cur = T;
    while (const auto *AT = Cur->getAs<AttributedType>()) {
      switch (AT->getAttrKind()) {
      case attr::HLSLRowMajor:
        return true;
      case attr::HLSLColumnMajor:
        return false;
      default:
        break;
      }
      Cur = AT->getModifiedType();
    }
  }
  return LangOpts.getDefaultMatrixMemoryLayout() ==
         LangOptions::MatrixMemoryLayout::MatrixRowMajor;
}

/// Returns true if matrices of \p T should be laid out in column-major order.
/// Mirrors `isMatrixRowMajor`; per-decl HLSL attributes win over the
/// `-fmatrix-memory-layout=` default.
inline bool isMatrixColumnMajor(const LangOptions &LangOpts, QualType T) {
  if (LangOpts.HLSL && !T.isNull()) {
    QualType Cur = T;
    while (const auto *AT = Cur->getAs<AttributedType>()) {
      switch (AT->getAttrKind()) {
      case attr::HLSLColumnMajor:
        return true;
      case attr::HLSLRowMajor:
        return false;
      default:
        break;
      }
      Cur = AT->getModifiedType();
    }
  }
  return LangOpts.getDefaultMatrixMemoryLayout() ==
         LangOptions::MatrixMemoryLayout::MatrixColMajor;
}
} // namespace clang

#endif // LLVM_CLANG_AST_MATRIXUTILS_H
