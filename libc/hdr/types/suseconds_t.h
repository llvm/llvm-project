//===-- Proxy for suseconds_t ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_HDR_TIMES_SUSECONDS_T_H
#define LLVM_LIBC_HDR_TIMES_SUSECONDS_T_H

#ifdef LIBC_FULL_BUILD

#include "include/llvm-libc-types/suseconds_t.h"

#else // Overlay mode

#include <sys/types.h>

#endif // LLVM_LIBC_FULL_BUILD

#endif // #ifndef LLVM_LIBC_HDR_TIMES_SUSECONDS_T_H
