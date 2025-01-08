//===-- Unittests for fromfpx ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FromfpxTest.h"

#include "src/math/fromfpx.h"

LIST_FROMFPX_TESTS(double, LIBC_NAMESPACE::fromfpx)
