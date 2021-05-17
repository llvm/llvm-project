//===------------------- uncaught_exceptions.pass.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// __cxa_uncaught_exceptions is not re-exported from libc++ until macOS 10.15.
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.14
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.13
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.12
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9

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
