//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER
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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

struct alignas(32) A {
    int field;
};

int main(int, char**)
{
    std::pair<A*, std::ptrdiff_t> ip = std::get_temporary_buffer<A>(5);
    assert(!(ip.first == nullptr) ^ (ip.second == 0));
    assert(reinterpret_cast<std::uintptr_t>(ip.first) % alignof(A) == 0);
    std::return_temporary_buffer(ip.first);

  return 0;
}
