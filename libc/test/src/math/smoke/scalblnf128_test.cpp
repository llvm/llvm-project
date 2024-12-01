//===-- Unittests for scalblnf128 -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScalbnTest.h"

#include "src/math/scalblnf128.h"

LIST_SCALBN_TESTS(float128, long, LIBC_NAMESPACE::scalblnf128)
