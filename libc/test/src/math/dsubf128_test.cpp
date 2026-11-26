//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for dsubf128.
///
//===----------------------------------------------------------------------===//

#include "SubTest.h"

#include "src/__support/FPUtil/float128.h"
#include "src/math/dsubf128.h"

LIST_SUB_TESTS(Dsubf128, double, float128, LIBC_NAMESPACE::dsubf128)
