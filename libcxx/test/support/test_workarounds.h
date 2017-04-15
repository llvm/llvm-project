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
#endif

// FIXME(EricWF): Remove this. This macro guards tests for upcoming changes
// and fixes to unique_ptr. Until the changes have been implemented in trunk
// the tests have to be disabled. However the tests have been left in until
// then so they can be used by other standard libraries.
#if defined(_LIBCPP_VERSION)
# define TEST_WORKAROUND_UPCOMING_UNIQUE_PTR_CHANGES
#endif

#endif // SUPPORT_TEST_WORKAROUNDS_H
