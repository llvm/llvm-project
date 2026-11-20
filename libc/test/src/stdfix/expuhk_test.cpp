//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for expuhk.
///
//===----------------------------------------------------------------------===//

#include "test/src/stdfix/ExpTest.h"

#include "src/stdfix/expuhk.h"

LIST_EXP_TESTS(uhk, unsigned short accum, LIBC_NAMESPACE::expuhk);
