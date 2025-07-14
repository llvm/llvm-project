//===-- Unittests for wcstoul ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcstoul.h"

#include "test/UnitTest/Test.h"

#include "WcstolTest.h"

WCSTOL_TEST(Wcstoul, LIBC_NAMESPACE::wcstoul)