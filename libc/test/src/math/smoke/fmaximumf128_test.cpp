//===-- Unittests for fmaximumf128-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMaximumTest.h"

#include "src/math/fmaximumf128.h"

LIST_FMAXIMUM_TESTS(float128, LIBC_NAMESPACE::fmaximumf128)
