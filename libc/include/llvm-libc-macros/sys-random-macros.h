//===-- Macros defined in sys/random.h header file ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_SYS_RANDOM_MACROS_H
#define __LLVM_LIBC_MACROS_SYS_RANDOM_MACROS_H

#ifdef __unix__
#include "linux/sys-random-macros.h"
#endif

#endif // __LLVM_LIBC_MACROS_SYS_RANDOM_MACROS_H
