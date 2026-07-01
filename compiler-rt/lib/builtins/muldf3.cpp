//===-- lib/muldf3.cpp - Quad-precision addition (libc-backed) --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// __muldf3 implemented on top of LLVM-libc's shared::muldf3 instruction.
//
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/muldf3.h"

extern "C" COMPILER_RT_ABI fp_t __muldf3(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::muldf3(a, b);
}
