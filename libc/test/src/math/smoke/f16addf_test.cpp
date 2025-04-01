//===-- Unittests for f16addf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AddTest.h"

#include "src/math/f16addf.h"

LIST_ADD_TESTS(float16, float, LIBC_NAMESPACE::f16addf)
