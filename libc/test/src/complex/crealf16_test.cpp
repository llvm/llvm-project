//===-- Unittests for crealf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CRealTest.h"

#include "src/complex/crealf16.h"

#if defined(LIBC_TYPES_HAS_CFLOAT16)

LIST_CREAL_TESTS(cfloat16, float16, LIBC_NAMESPACE::crealf16)

#endif // LIBC_TYPES_HAS_CFLOAT16
