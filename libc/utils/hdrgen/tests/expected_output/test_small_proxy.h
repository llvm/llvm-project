//===-- Implementation proxy header for <test_small.h> --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_LIBC_HDR_TEST_SMALL_PROXY_H
#define LLVM_LIBC_HDR_TEST_SMALL_PROXY_H

#ifdef LIBC_FULL_BUILD

#include "llvm-libc-macros/CONST_FUNC_A.h"
#include "llvm-libc-macros/test_more-macros.h"
#include "llvm-libc-macros/test_small-macros.h"
#include "llvm-libc-types/float128.h"
#include "llvm-libc-types/type_a.h"
#include "llvm-libc-types/type_b.h"

#else // Overlay mode

#include <test_small.h>

#endif // LLVM_LIBC_FULL_BUILD

#endif // LLVM_LIBC_HDR_TEST_SMALL_PROXY_H
