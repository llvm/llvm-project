//===-- Unittests for conjf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConjTest.h"

#include "src/complex/conjf16.h"

LIST_CONJ_TESTS(cfloat16, float16, LIBC_NAMESPACE::conjf16)
