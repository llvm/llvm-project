//===------------------ InterpBuiltinBitCast.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_BUILITN_BIT_CAST_H
#define LLVM_CLANG_AST_INTERP_BUILITN_BIT_CAST_H

#include <cstddef>

namespace clang {
namespace interp {
class Pointer;
class InterpState;
class CodePtr;

bool DoBitCast(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               std::byte *Buff, size_t BuffSize, bool &HasIndeterminateBits);

} // namespace interp
} // namespace clang

#endif
