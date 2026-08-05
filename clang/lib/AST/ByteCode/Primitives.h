//===------ Primitives.h - Types for the constexpr VM -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities and helper functions for all primitive types:
//  - Integral
//  - Floating
//  - Boolean
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_PRIMITIVES_H
#define LLVM_CLANG_AST_INTERP_PRIMITIVES_H

#include "clang/AST/ComparisonCategories.h"

namespace clang {
namespace interp {

enum class IntegralKind : uint8_t {
  /// Just a number, nothing else.
  Number = 0,
  /// A pointer to a ValueDecl.
  Address,
  /// A pointer to an interp::Block.
  BlockAddress,
  /// A pointer to a AddrLabelExpr.
  LabelAddress,
  /// A pointer to a FunctionDecl.
  FunctionAddress,
  /// Difference between two AddrLabelExpr.
  AddrLabelDiff
};

/// Helper to compare two comparable types.
template <typename T> ComparisonCategoryResult Compare(const T &X, const T &Y) {
  if (X < Y)
    return ComparisonCategoryResult::Less;
  if (X > Y)
    return ComparisonCategoryResult::Greater;
  return ComparisonCategoryResult::Equal;
}

template <typename T> inline bool CheckAddUB(T A, T B, T &R) {
  if constexpr (std::is_signed_v<T>) {
    return llvm::AddOverflow<T>(A, B, R);
  } else {
    R = A + B;
    return false;
  }
}

template <typename T> inline bool CheckSubUB(T A, T B, T &R) {
  if constexpr (std::is_signed_v<T>) {
    return llvm::SubOverflow<T>(A, B, R);
  } else {
    R = A - B;
    return false;
  }
}

template <typename T> inline bool CheckMulUB(T A, T B, T &R) {
  if constexpr (std::is_signed_v<T>) {
    return llvm::MulOverflow<T>(A, B, R);
  } else if constexpr (sizeof(T) < sizeof(int)) {
    // Silly integer promotion rules will convert both A and B to int,
    // even it T is unsigned. Prevent that by manually casting to uint first.
    R = static_cast<T>(static_cast<unsigned>(A) * static_cast<unsigned>(B));
    return false;
  } else {
    R = A * B;
    return false;
  }
}

} // namespace interp
} // namespace clang

#endif
