//===-- Implementation header for unlink ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_UNLINK_H
#define LLVM_LIBC_SRC_UNISTD_UNLINK_H

namespace __llvm_libc {

int unlink(const char *path);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_UNISTD_UNLINK_H
