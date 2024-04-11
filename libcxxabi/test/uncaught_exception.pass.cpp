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

// __cxa_uncaught_exception was not re-exported from libc++ previously. This leads
// to undefined symbols when linking against a libc++ that re-exports the symbols,
// but running against a libc++ that doesn't. Fortunately, usage of __cxa_uncaught_exception()
// in the wild seems to be close to non-existent.
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx{{(11|12|13|14)([.][0-9]+)?}}

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
