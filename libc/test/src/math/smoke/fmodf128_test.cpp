//===-- Unittests for fmodf128 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FModTest.h"

#include "src/math/fmodf128.h"

LIST_FMOD_TESTS(float128, LIBC_NAMESPACE::fmodf128)
