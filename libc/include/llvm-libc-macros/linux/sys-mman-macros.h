//===-- Definition of macros from sys/mman.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_LINUX_SYS_MMAN_MACROS_H
#define __LLVM_LIBC_MACROS_LINUX_SYS_MMAN_MACROS_H

// Use definitions from <linux/mman.h> to dispatch arch-specific flag values.
// For example, MCL_CURRENT/MCL_FUTURE/MCL_ONFAULT are different on different
// architectures.
#include <linux/mman.h>

// Some posix standard flags are not defined in linux/mman.h.
// Posix mmap flags.
#define MAP_FAILED ((void *)-1)
// Posix memory advise flags. (posix_madvise)
#define POSIX_MADV_NORMAL MADV_NORMAL
#define POSIX_MADV_SEQUENTIAL MADV_SEQUENTIAL
#define POSIX_MADV_RANDOM MADV_RANDOM
#define POSIX_MADV_WILLNEED MADV_WILLNEED
#define POSIX_MADV_DONTNEED MADV_DONTNEED
#endif // __LLVM_LIBC_MACROS_LINUX_SYS_MMAN_MACROS_H
