//===-- Implementation header for _exit -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD__EXIT_H
#define LLVM_LIBC_SRC_UNISTD__EXIT_H

namespace LIBC_NAMESPACE {

[[noreturn]] void _exit(int status);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_UNISTD__EXIT_H
