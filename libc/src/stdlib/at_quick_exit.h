//===-- Implementation header for at_quick_exit -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_AT_QUICK_EXIT_H
#define LLVM_LIBC_SRC_STDLIB_AT_QUICK_EXIT_H

#include "hdr/types/atexithandler_t.h"

namespace LIBC_NAMESPACE {

int at_quick_exit(__atexithandler_t);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_AT_QUICK_EXIT_H
