//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux specific declarations of macros from sys/shm.h.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_LINUX_SYS_SHM_MACROS_H
#define LLVM_LIBC_MACROS_LINUX_SYS_SHM_MACROS_H

#include "../../__llvm-libc-common.h"
#include "../stdint-macros.h"

// shmat flags
#define SHM_RDONLY 010000
#define SHM_RND 020000
#define SHM_REMAP 040000
#define SHM_EXEC 0100000

#define SHMLBA 4096
#define SHM_FAILED                                                             \
  __LLVM_LIBC_CAST(reinterpret_cast, void *,                                   \
                   __LLVM_LIBC_CAST(static_cast, intptr_t, -1))

#endif // LLVM_LIBC_MACROS_LINUX_SYS_SHM_MACROS_H
