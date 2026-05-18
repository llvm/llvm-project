//===-- Unittests for modfbf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModfTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/modfbf16.h"

LIST_MODF_TESTS(bfloat16, LIBC_NAMESPACE::modfbf16)
