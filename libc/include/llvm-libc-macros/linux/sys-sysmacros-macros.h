//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux specific declarations of macros from sys/sysmacros.h.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_LINUX_SYS_SYSMACROS_MACROS_H
#define LLVM_LIBC_MACROS_LINUX_SYS_SYSMACROS_MACROS_H

#include "../../llvm-libc-types/dev_t.h"

// A device number holds the two parts split apart and interleaved: twelve bits
// of the major number and eight of the minor sit low, and the rest of each
// sits high.
//
// POSIX allows these to be macros and every library makes them so. What
// matters is that the expansion names none of them: a caller with a variable
// of its own called major still reaches the right thing, which a macro naming
// the function straight back cannot manage.
#define major(dev)                                                             \
  ((unsigned int)((((__UINT64_TYPE__)(dev)) >> 8) & 0x00000fff) |              \
   (unsigned int)((((__UINT64_TYPE__)(dev)) >> 32) & 0xfffff000))

#define minor(dev)                                                             \
  ((unsigned int)(((__UINT64_TYPE__)(dev)) & 0x000000ff) |                     \
   (unsigned int)((((__UINT64_TYPE__)(dev)) >> 12) & 0xffffff00))

#define makedev(maj, min)                                                      \
  ((dev_t)(((((__UINT64_TYPE__)(maj)) & 0x00000fff) << 8) |                    \
           ((((__UINT64_TYPE__)(maj)) & 0xfffff000) << 32) |                   \
           (((__UINT64_TYPE__)(min)) & 0x000000ff) |                           \
           ((((__UINT64_TYPE__)(min)) & 0xffffff00) << 12)))

#endif // LLVM_LIBC_MACROS_LINUX_SYS_SYSMACROS_MACROS_H
