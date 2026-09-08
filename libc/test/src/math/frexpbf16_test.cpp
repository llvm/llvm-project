//===-- Unittests for frexpbf16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains tests for frexpbf16
///
//===----------------------------------------------------------------------===//

#include "FrexpTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/frexpbf16.h"

LIST_FREXP_TESTS(bfloat16, LIBC_NAMESPACE::frexpbf16)
