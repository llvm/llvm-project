//===-- Unittests for fminimum_magf128-------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMinimumMagTest.h"

#include "src/__support/FPUtil/float128.h"
#include "src/math/fminimum_magf128.h"

#ifndef LIBC_TYPES_HAS_NATIVE_FLOAT128
using float128 = LIBC_NAMESPACE::fputil::Float128;
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128

LIST_FMINIMUM_MAG_TESTS(float128, LIBC_NAMESPACE::fminimum_magf128)
