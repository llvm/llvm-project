// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_WORKAROUNDS_H
#define SUPPORT_TEST_WORKAROUNDS_H

#include "test_macros.h"

#if defined(TEST_COMPILER_C1XX)
# define TEST_WORKAROUND_C1XX_BROKEN_NULLPTR_CONVERSION_OPERATOR
# define TEST_WORKAROUND_C1XX_BROKEN_IS_TRIVIALLY_COPYABLE
#endif

#endif // SUPPORT_TEST_WORKAROUNDS_H
