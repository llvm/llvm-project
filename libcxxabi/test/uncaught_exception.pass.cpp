//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// This tests that libc++abi still provides __cxa_uncaught_exception() for
// ABI compatibility, even though the Standard doesn't require it to.

#include <cxxabi.h>
#include <cassert>

// namespace __cxxabiv1 {
//      extern bool __cxa_uncaught_exception () throw();
// }

struct A {
    ~A() { assert( __cxxabiv1::__cxa_uncaught_exception()); }
};

int main () {
    try { A a; throw 3; assert(false); }
    catch (int) {}
}
