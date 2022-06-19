//===-- Implementation header for unlinkat ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_UNLINKAT_H
#define LLVM_LIBC_SRC_UNISTD_UNLINKAT_H

namespace __llvm_libc {

int unlinkat(int dfd, const char *path, int flags);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_UNISTD_UNLINKAT_H
