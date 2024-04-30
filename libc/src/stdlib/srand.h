//===-- Implementation header for srand -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_SRAND_H
#define LLVM_LIBC_SRC_STDLIB_SRAND_H

#include <stdlib.h>

namespace LIBC_NAMESPACE {

void srand(unsigned int seed);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_SRAND_H
