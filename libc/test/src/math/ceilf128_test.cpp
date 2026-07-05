//===-- Unittests for ceilf128 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CeilTest.h"

#include "src/__support/FPUtil/float128.h"
#include "src/math/ceilf128.h"

using LIBC_NAMESPACE::fputil::Float128;

LIST_CEIL_TESTS(Float128, LIBC_NAMESPACE::ceilf128)
