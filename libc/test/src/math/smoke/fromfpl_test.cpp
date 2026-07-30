//===-- Unittests for fromfpl ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FromfpTest.h"

#include "src/math/fromfpl.h"

LIST_FROMFP_TESTS(long double, LIBC_NAMESPACE::fromfpl)
