//===-- Unittests for ufromfpxf16 -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UfromfpxTest.h"

#include "src/math/ufromfpxf16.h"

LIST_UFROMFPX_TESTS(float16, LIBC_NAMESPACE::ufromfpxf16)
