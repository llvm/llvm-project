//===-- Utilities for getting secure randomness -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_RANDOMNESS_H
#define LLVM_LIBC_SRC___SUPPORT_RANDOMNESS_H

#include "src/__support/common.h"

#define __need_size_t
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
void random_fill(void *buf, unsigned long size);
} // namespace LIBC_NAMESPACE_DECL
#endif // LLVM_LIBC_SRC___SUPPORT_RANDOMNESS_H
