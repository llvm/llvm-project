//===-- Implementation header for atexit ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ATEXIT_H
#define LLVM_LIBC_SRC_STDLIB_ATEXIT_H

#include <stddef.h> // For size_t

namespace LIBC_NAMESPACE {

constexpr size_t CALLBACK_LIST_SIZE_FOR_TESTS = 1024;

int atexit(void (*function)());

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_ATEXIT_H
