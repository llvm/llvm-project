//===-- Unittests for remquof16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains tests for remquof16
///
//===----------------------------------------------------------------------===//

#include "RemQuoTest.h"

#include "src/math/remquof16.h"

LIST_REMQUO_TESTS(float16, LIBC_NAMESPACE::remquof16)
