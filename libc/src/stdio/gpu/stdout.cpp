//===-- Definition of the global stdout object ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

namespace __llvm_libc {
static struct {
} stub;
FILE *stdout = reinterpret_cast<FILE *>(&stub);
} // namespace __llvm_libc
extern "C" FILE *stdout = reinterpret_cast<FILE *>(&__llvm_libc::stub);
