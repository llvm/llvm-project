//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// [[nodiscard]] void* operator new(std::size_t);
// [[nodiscard]] void* operator new(std::size_t, std::nothrow_t const&);
// [[nodiscard]] void* operator new(std::size_t, std::align_val_t);
// [[nodiscard]] void* operator new(std::size_t, std::align_val_t, std::nothrow_t const&);

// [[nodiscard]] is not supported at all in c++03
// UNSUPPORTED: c++03

// [[nodiscard]] enabled before C++20 in libc++ as an extension
// UNSUPPORTED: (c++11 || c++14 || c++17) && !stdlib=libc++

// Libc++ when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

#include <new>

#include "test_macros.h"

void f() {
    ::operator new(4); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new(4, std::nothrow); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 17
    ::operator new(4, std::align_val_t{4});  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new(4, std::align_val_t{4}, std::nothrow);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
