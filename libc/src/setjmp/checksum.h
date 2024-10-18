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
extern __UINT64_TYPE__ checksum_cookie;

// abitrary prime number
LIBC_INLINE constexpr __UINT64_TYPE__ ROTATION = 13;
void initialize();
extern "C" [[gnu::cold, noreturn]] void __libc_jmpbuf_corruption();
} // namespace jmpbuf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SETJMP_CHECKSUM_H
