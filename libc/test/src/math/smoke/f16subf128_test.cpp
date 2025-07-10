//===-- Unittests for f16subf128 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubTest.h"

#include "src/math/f16subf128.h"

LIST_SUB_TESTS(float16, float128, LIBC_NAMESPACE::f16subf128)
