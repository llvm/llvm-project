//===-- 128-bit signed and unsigned int types -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_UINT128_H
#define LLVM_LIBC_SRC_SUPPORT_UINT128_H

#include "UInt.h"

#if !defined(__SIZEOF_INT128__)
using UInt128 = __llvm_libc::cpp::UInt<128>;
using Int128 = __llvm_libc::cpp::Int<128>;
#else
using UInt128 = __uint128_t;
using Int128 = __int128_t;
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_UINT128_H
