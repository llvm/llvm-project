//===-- lib/fixunssfti.cpp - libc-backed __fixunssfti -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// __fixunssfti implemented on top of LLVM-libc's shared::fixunssfti.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_128BIT
#define SINGLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/fixunssfti.h"

extern "C" COMPILER_RT_ABI tu_int __fixunssfti(fp_t a) {
  return LIBC_NAMESPACE::shared::fixunssfti(a);
}

#endif // CRT_HAS_128BIT
