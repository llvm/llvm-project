//===-- Implementation header for jmpbuf checksum ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SETJMP_CHECKSUM_H
#define LLVM_LIBC_SRC_SETJMP_CHECKSUM_H

#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {
namespace jmpbuf {

extern __UINTPTR_TYPE__ value_mask;
extern __UINTPTR_TYPE__ checksum_cookie;

// single register update derived from aHash
// https://github.com/tkaitchuck/aHash/blob/master/src/fallback_hash.rs#L95
//
// checksum = folded_multiple(data ^ checksum, MULTIPLE)
// folded_multiple(x, m) = HIGH(x * m) ^ LOW(x * m)

// From Knuth's PRNG
LIBC_INLINE constexpr __UINTPTR_TYPE__ MULTIPLE =
    static_cast<__UINTPTR_TYPE__>(6364136223846793005ull);
void initialize();
extern "C" [[gnu::cold, noreturn]] void __libc_jmpbuf_corruption();
} // namespace jmpbuf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SETJMP_CHECKSUM_H
