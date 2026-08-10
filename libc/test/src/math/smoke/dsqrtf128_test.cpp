//===-- Unittests for dsqrtf128 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SqrtTest.h"

#include "src/math/dsqrtf128.h"

LIST_NARROWING_SQRT_TESTS(double, float128, LIBC_NAMESPACE::dsqrtf128)
