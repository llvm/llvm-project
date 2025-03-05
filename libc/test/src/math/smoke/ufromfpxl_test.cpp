//===-- Unittests for ufromfpxl -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UfromfpxTest.h"

#include "src/math/ufromfpxl.h"

LIST_UFROMFPX_TESTS(long double, LIBC_NAMESPACE::ufromfpxl)
