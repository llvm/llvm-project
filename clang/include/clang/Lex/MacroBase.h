//===--- MacroBase.h - Forward declarations for macros ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Forward-declares types that need PointerLikeTypeTraits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_MACROBASE_H
#define LLVM_CLANG_LEX_MACROBASE_H

#include "llvm/Support/PointerLikeTypeTraits.h"

namespace clang {
class MacroInfo;
} // namespace clang

namespace llvm {
template <> struct PointerLikeTypeTraits<::clang::MacroInfo *> {
  static inline void *getAsVoidPointer(::clang::MacroInfo *P) { return P; }
  static inline ::clang::MacroInfo *getFromVoidPointer(void *P) {
    return static_cast<::clang::MacroInfo *>(P);
  }
  static constexpr int NumLowBitsAvailable = 2;
};

template <> struct PointerLikeTypeTraits<const ::clang::MacroInfo *> {
  static inline const void *getAsVoidPointer(const ::clang::MacroInfo *P) {
    return P;
  }
  static inline const ::clang::MacroInfo *getFromVoidPointer(const void *P) {
    return static_cast<const ::clang::MacroInfo *>(P);
  }
  static constexpr int NumLowBitsAvailable = 2;
};
} // namespace llvm

#endif // LLVM_CLANG_LEX_MACROBASE_H
