//===-- Definition of sigset_t type ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_SIGSET_T_H__
#define __LLVM_LIBC_TYPES_SIGSET_T_H__

#include <llvm-libc-macros/signal-macros.h>

// This definition can be adjusted/specialized for different targets and
// platforms as necessary. This definition works for Linux on most targets.
typedef struct {
  unsigned long __signals[__NSIGSET_WORDS];
} sigset_t;

#endif // __LLVM_LIBC_TYPES_SIGSET_T_H__
