//===-- Unittests for conjf128 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConjTest.h"

#include "src/complex/conjf128.h"

LIST_CONJ_TESTS(cfloat128, float128, LIBC_NAMESPACE::conjf128)
