//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for wcstoimax.
///
//===----------------------------------------------------------------------===//

#include "src/inttypes/wcstoimax.h"

#include "test/UnitTest/Test.h"

#include "test/src/wchar/WcstolTest.h"

WCSTOL_TEST(Wcstoimax, LIBC_NAMESPACE::wcstoimax)
