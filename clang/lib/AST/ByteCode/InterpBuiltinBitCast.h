//===------------------ InterpBuiltinBitCast.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_BUILTIN_BIT_CAST_H
#define LLVM_CLANG_AST_INTERP_BUILTIN_BIT_CAST_H

#include "BitcastBuffer.h"
#include <cstddef>

namespace clang {
namespace interp {
class Pointer;
class InterpState;
class CodePtr;
class Context;

inline static void swapBytes(std::byte *M, size_t N) {
  for (size_t I = 0; I != (N / 2); ++I)
    std::swap(M[I], M[N - 1 - I]);
}

bool DoBitCast(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               std::byte *Buff, Bits BitWidth, Bits FullBitWidth,
               bool &HasIndeterminateBits);
bool DoBitCastPtr(InterpState &S, CodePtr OpPC, const Pointer &FromPtr,
                  Pointer &ToPtr);
bool DoBitCastPtr(InterpState &S, CodePtr OpPC, const Pointer &FromPtr,
                  Pointer &ToPtr, size_t Size);
bool readPointerToBuffer(const Context &Ctx, const Pointer &FromPtr,
                         BitcastBuffer &Buffer, bool ReturnOnUninit);

bool DoMemcpy(InterpState &S, CodePtr OpPC, const Pointer &SrcPtr,
              const Pointer &DestPtr, Bits Size);

} // namespace interp
} // namespace clang

#endif
