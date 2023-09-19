//===-- Unittests for nextafterl ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NextAfterTest.h"

#include "src/math/nextafterl.h"

LIST_NEXTAFTER_TESTS(long double, __llvm_libc::nextafterl)
