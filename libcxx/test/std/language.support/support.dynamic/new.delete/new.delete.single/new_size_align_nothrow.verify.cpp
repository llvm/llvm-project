//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <new>

// void* operator new(std::size_t, std::align_val_t, std::nothrow_t &);

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Libcxx when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

// REQUIRES: -faligned-allocation
// ADDITIONAL_COMPILE_FLAGS: -faligned-allocation

#include <new>

int main(int, char**) {
    ::operator new(4, std::align_val_t{4}, std::nothrow);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    return 0;
}
