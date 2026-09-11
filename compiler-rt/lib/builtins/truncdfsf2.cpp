//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __truncdfsf2, truncate double to float,
/// on top of LLVM-libc's shared::truncdfsf2.
///
//===----------------------------------------------------------------------===//

#include "shared/builtins/truncdfsf2.h"
#include "fp_libc_config.h"
#include "int_lib.h"

extern "C" COMPILER_RT_ABI float __truncdfsf2(double a) {
  return LIBC_NAMESPACE::shared::truncdfsf2(a);
}
