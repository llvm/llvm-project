//===-- Implementation header for strtoul -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_STRTOUL_H
#define LLVM_LIBC_SRC_STDLIB_STRTOUL_H

namespace LIBC_NAMESPACE {

unsigned long strtoul(const char *__restrict str, char **__restrict str_end,
                      int base);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_STRTOUL_H
