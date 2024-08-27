//===-- Unittests for fmul ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MulTest.h"

#include "src/math/fmul.h"

LIST_MUL_TESTS(float, double, LIBC_NAMESPACE::fmul)
