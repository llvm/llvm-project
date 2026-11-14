//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for iscanonicalf80.
///
//===----------------------------------------------------------------------===//

#include "IsCanonicalTest.h"

#include "src/__support/FPUtil/float80.h"
#include "src/math/iscanonicalf80.h"

using LIBC_NAMESPACE::fputil::Float80;

LIST_ISCANONICAL_TESTS(Iscanonicalf80, Float80, LIBC_NAMESPACE::iscanonicalf80)
