//===-- Unittests for crealf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CRealTest.h"

#include "src/complex/crealf.h"

LIST_CREAL_TESTS(_Complex float, float, LIBC_NAMESPACE::crealf)
