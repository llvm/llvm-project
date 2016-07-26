// -*- C++ -*-
//===---------------------------- test_macros.h ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_THROW_H
#define SUPPORT_TEST_THROW_H

#include "test_macros.h"
#include <cstdlib>

template <class Ex>
TEST_NORETURN
inline void test_throw() {
#ifndef TEST_HAS_NO_EXCEPTIONS
       throw Ex();
#else
       std::abort();
#endif
}

#endif // SUPPORT_TEST_THROW_H
