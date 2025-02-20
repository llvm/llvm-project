//===-- Unittests for cimag -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CImagTest.h"

#include "src/complex/cimag.h"

LIST_CIMAG_TESTS(_Complex double, double, LIBC_NAMESPACE::cimag)
