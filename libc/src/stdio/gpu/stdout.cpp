//===-- Definition of the global stdout object ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

namespace LIBC_NAMESPACE {
static struct {
} stub;
FILE *stdout = reinterpret_cast<FILE *>(&stub);
} // namespace LIBC_NAMESPACE
extern "C" FILE *stdout = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::stub);
