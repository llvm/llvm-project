//===-- Unittests for cproj -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CprojTest.h"

#include "src/complex/cproj.h"

LIST_CPROJ_TESTS(_Complex double, double, LIBC_NAMESPACE::cproj)
