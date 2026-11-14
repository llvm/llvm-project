//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of loff_t type for Linux.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_LINUX_LOFF_T_H
#define LLVM_LIBC_TYPES_LINUX_LOFF_T_H

#include "../off_t.h"

// The name for an offset which is wide whatever the machine. off_t here is
// already that, and the two have to be the same type or a call which takes
// one will not take a pointer to the other. glibc makes them the same too.
typedef off_t loff_t;

#endif // LLVM_LIBC_TYPES_LINUX_LOFF_T_H
