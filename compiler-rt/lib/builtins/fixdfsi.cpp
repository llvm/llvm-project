//===-- lib/fixdfsi.cpp - libc-backed __fixdfsi -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// __fixdfsi implemented on top of LLVM-libc's shared::fixdfsi.
//
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/fixdfsi.h"

extern "C" COMPILER_RT_ABI si_int __fixdfsi(fp_t a) {
  return LIBC_NAMESPACE::shared::fixdfsi(a);
}
