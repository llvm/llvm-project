//===-- Unittests for nextafterf ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NextAfterTest.h"

#include "src/math/nextafterf.h"

LIST_NEXTAFTER_TESTS(float, __llvm_libc::nextafterf)
