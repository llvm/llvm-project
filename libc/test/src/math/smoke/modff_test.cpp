//===-- Unittests for modff -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModfTest.h"

#include "src/math/modff.h"

LIST_MODF_TESTS(float, __llvm_libc::modff)
