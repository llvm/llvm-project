//===-- Implementation header for shm_unlink function ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MMAN_SHM_UNLINK_H
#define LLVM_LIBC_SRC_SYS_MMAN_SHM_UNLINK_H

namespace LIBC_NAMESPACE {

int shm_unlink(const char *name);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_MMAN_SHM_UNLINK_H
