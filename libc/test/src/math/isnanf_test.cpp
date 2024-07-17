//===-- Unittests for isnanf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IsNanTest.h"

#include "src/math/isnanf.h"

LIST_ISNAN_TESTS(float, LIBC_NAMESPACE::isnanf)
