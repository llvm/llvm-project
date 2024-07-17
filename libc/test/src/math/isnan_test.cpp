//===-- Unittests for isnan -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IsNanTest.h"

// We need to avoid expanding isnan to __builtin_isnan.
#undef isnan

#include "src/math/isnan.h"

LIST_ISNAN_TESTS(double, LIBC_NAMESPACE::isnan)
