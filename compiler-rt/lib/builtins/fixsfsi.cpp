//===-- lib/fixsfsi.cpp - libc-backed __fixsfsi -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// __fixsfsi implemented on top of LLVM-libc's shared::fixsfsi.
//
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/fixsfsi.h"

extern "C" COMPILER_RT_ABI si_int __fixsfsi(fp_t a) {
  return LIBC_NAMESPACE::shared::fixsfsi(a);
}
