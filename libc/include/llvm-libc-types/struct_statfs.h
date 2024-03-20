//===-- Definition of type struct statfs ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_STRUCT_STATFS_H
#define LLVM_LIBC_TYPES_STRUCT_STATFS_H

// Statfs is not specified by POSIX, rather it is OS specific. So, we include
// UAPI header to provide its definition.

#ifdef __linux__
#include <asm/statfs.h>
#endif

#endif // LLVM_LIBC_TYPES_STRUCT_STATFS_H
