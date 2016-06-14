//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// TODO(EricWF) Investigate why typeid(...).name() returns a different string
// on GCC 4.9 but not newer GCCs.
// XFAIL: gcc-4.9

// THIS TESTS C++03 EXTENSIONS.

// <mutex>

// template <class ...Mutex> class lock_guard;

// Test that the the variadic lock guard implementation mangles the same in
// C++11 and C++03. This is important since the mangling of `lock_guard` depends
// on it being declared as a variadic template, even in C++03.

#define _LIBCPP_ABI_VARIADIC_LOCK_GUARD
#include <mutex>
#include <string>
#include <typeinfo>
#include <cassert>

int main() {
    const std::string expect = "NSt3__110lock_guardIJNS_5mutexEEEE";
    assert(typeid(std::lock_guard<std::mutex>).name() == expect);
}
