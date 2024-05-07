//===-- Definition of macros from fenv.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_HDR_FENV_MACROS_H
#define LLVM_LIBC_HDR_FENV_MACROS_H

#ifdef LIBC_FULL_BUILD

#include "include/llvm-libc-macros/fenv-macros.h"

#else // Overlay mode

#include <fenv.h>

// If this is not provided by the system, define it for use internally.
#ifndef __FE_DENORM
#define __FE_DENORM (1 << 6)
#endif

#endif // LLVM_LIBC_FULL_BUILD

#endif // LLVM_LIBC_HDR_FENV_MACROS_H
