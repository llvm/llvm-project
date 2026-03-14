//===-- Unittests for scalblnl --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScalbnTest.h"

#include "src/math/scalblnl.h"

LIST_SCALBN_TESTS(long double, long, LIBC_NAMESPACE::scalblnl)
