//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// __cxa_uncaught_exceptions is not re-exported from libc++ until macOS 10.15.
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

#include <cxxabi.h>
#include <cassert>

// namespace __cxxabiv1 {
//      extern unsigned int __cxa_uncaught_exceptions() throw();
// }

struct A {
    A(unsigned cnt) : data_(cnt) {}
    ~A() { assert( data_ == __cxxabiv1::__cxa_uncaught_exceptions()); }
    unsigned data_;
};

int main () {
    try { A a(1); throw 3; assert(false); }
    catch (int) {}
}
