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

#if defined(TEST_COMPILER_EDG)
# define TEST_WORKAROUND_EDG_EXPLICIT_CONSTEXPR
#endif

#if defined(TEST_COMPILER_C1XX)
# define TEST_WORKAROUND_C1XX_BROKEN_NULLPTR_CONVERSION_OPERATOR
# define TEST_WORKAROUND_C1XX_BROKEN_IS_TRIVIALLY_COPYABLE
# ifndef _MSC_EXTENSIONS
#  define TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
# endif
#endif

#endif // SUPPORT_TEST_WORKAROUNDS_H
