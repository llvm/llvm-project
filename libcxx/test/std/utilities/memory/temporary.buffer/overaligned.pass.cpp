//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Aligned allocations are not supported on macOS < 10.13
// Note: use 'unsupported' instead of 'xfail' to ensure
// we won't pass prior to c++17.
// UNSUPPORTED: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <memory>

// template <class T>
//   pair<T*, ptrdiff_t>
//   get_temporary_buffer(ptrdiff_t n);
//
// template <class T>
//   void
//   return_temporary_buffer(T* p);

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "test_macros.h"

struct alignas(32) A {
    int field;
};

int main(int, char**)
{
    std::pair<A*, std::ptrdiff_t> ip = std::get_temporary_buffer<A>(5);
    assert(!(ip.first == nullptr) ^ (ip.second == 0));
    assert(reinterpret_cast<uintptr_t>(ip.first) % alignof(A) == 0);
    std::return_temporary_buffer(ip.first);

  return 0;
}
