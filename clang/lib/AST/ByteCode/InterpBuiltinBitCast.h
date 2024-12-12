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

bool DoBitCast(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               std::byte *Buff, Bits BitWidth, Bits FullBitWidth,
               bool &HasIndeterminateBits);
bool DoBitCastPtr(InterpState &S, CodePtr OpPC, const Pointer &FromPtr,
                  Pointer &ToPtr);
bool DoBitCastPtr(InterpState &S, CodePtr OpPC, const Pointer &FromPtr,
                  Pointer &ToPtr, size_t Size);
bool readPointerToBuffer(const Context &Ctx, const Pointer &FromPtr,
                         BitcastBuffer &Buffer, bool ReturnOnUninit);
} // namespace interp
} // namespace clang

#endif
