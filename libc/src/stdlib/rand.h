//===-- Implementation header for rand --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_RAND_H
#define LLVM_LIBC_SRC_STDLIB_RAND_H

#include <stdlib.h>

namespace LIBC_NAMESPACE {

int rand(void);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_RAND_H
