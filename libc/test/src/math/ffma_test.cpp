//===-- Unittests for ffma ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FmaTest.h"

#include "src/math/ffma.h"

LIST_NARROWING_FMA_TESTS(float, double, LIBC_NAMESPACE::ffma)
