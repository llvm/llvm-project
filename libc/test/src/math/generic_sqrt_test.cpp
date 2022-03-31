//===-- Unittests for generic implementation of sqrt ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SqrtTest.h"

#include "src/__support/FPUtil/generic/sqrt.h"

LIST_SQRT_TESTS(double, __llvm_libc::fputil::sqrt<double>)
