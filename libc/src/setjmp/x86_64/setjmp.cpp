//===-- Implementation of setjmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/setjmp_impl.h"

#ifdef LIBC_TARGET_ARCH_IS_X86_64
#if LIBC_COPT_SETJMP_FORTIFICATION
#include "setjmp_64_fortified.cpp.inc"
#else
#include "setjmp_64.cpp.inc"
#endif
#else
#include "setjmp_32.cpp.inc"
#endif
