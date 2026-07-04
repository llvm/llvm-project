//===-- lib/fixsfdi.cpp - libc-backed __fixsfdi -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// __fixsfdi implemented on top of LLVM-libc's shared::fixsfdi.
//
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/fixsfdi.h"

extern "C" COMPILER_RT_ABI di_int __fixsfdi(fp_t a) {
  return LIBC_NAMESPACE::shared::fixsfdi(a);
}
