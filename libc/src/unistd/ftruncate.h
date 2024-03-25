//===-- Implementation header for ftruncate ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_FTRUNCATE_H
#define LLVM_LIBC_SRC_UNISTD_FTRUNCATE_H

#include <unistd.h>

namespace LIBC_NAMESPACE {

int ftruncate(int, off_t);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_UNISTD_FTRUNCATE_H
