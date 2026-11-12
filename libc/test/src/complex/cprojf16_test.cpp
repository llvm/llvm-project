//===-- Unittests for cprojf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CprojTest.h"

#include "src/complex/cprojf16.h"

LIST_CPROJ_TESTS(cfloat16, float16, LIBC_NAMESPACE::cprojf16)
