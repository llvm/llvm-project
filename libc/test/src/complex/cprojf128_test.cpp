//===-- Unittests for cprojf128 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CprojTest.h"

#include "src/complex/cprojf128.h"

#if defined(LIBC_TYPES_HAS_CFLOAT128)

LIST_CPROJ_TESTS(cfloat128, float128, LIBC_NAMESPACE::cprojf128)

#endif // LIBC_TYPES_HAS_CFLOAT128
