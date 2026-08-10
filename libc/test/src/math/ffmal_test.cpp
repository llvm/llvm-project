//===-- Unittests for ffmal -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FmaTest.h"

#include "src/math/ffmal.h"

LIST_NARROWING_FMA_TESTS(float, long double, LIBC_NAMESPACE::ffmal)
