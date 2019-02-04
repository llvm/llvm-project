//===----------------------- noexception4.pass.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: libcxxabi-no-exceptions

#include <cxxabi.h>
#include <exception>
#include <cassert>

// namespace __cxxabiv1 {
//      void *__cxa_current_primary_exception() throw();
//      extern bool          __cxa_uncaught_exception () throw();
//      extern unsigned int  __cxa_uncaught_exceptions() throw();
// }

int main ()
{
    // Trivially
    assert(nullptr == __cxxabiv1::__cxa_current_primary_exception());
    assert(!__cxxabiv1::__cxa_uncaught_exception());
    assert(0 == __cxxabiv1::__cxa_uncaught_exceptions());
    return 0;
}
