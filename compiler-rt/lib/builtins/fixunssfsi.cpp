//===-- lib/fixunssfsi.cpp - libc-backed __fixunssfsi -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// __fixunssfsi implemented on top of LLVM-libc's shared::fixunssfsi.
//
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/fixunssfsi.h"

extern "C" COMPILER_RT_ABI su_int __fixunssfsi(fp_t a) {
  return LIBC_NAMESPACE::shared::fixunssfsi(a);
}
