//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// template <class T>
//   pair<T*, ptrdiff_t>
//   get_temporary_buffer(ptrdiff_t n);
//
// template <class T>
//   void
//   return_temporary_buffer(T* p);

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

int main(int, char**)
{
    std::pair<int*, std::ptrdiff_t> ip = std::get_temporary_buffer<int>(5);
    assert(ip.first);
    assert(ip.second == 5);
    std::return_temporary_buffer(ip.first);

  return 0;
}
