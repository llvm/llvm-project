//===-- Unittests for scalbnf16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScalbnTest.h"

#include "src/math/scalbnf16.h"

LIST_SCALBN_TESTS(float16, int, LIBC_NAMESPACE::scalbnf16)
