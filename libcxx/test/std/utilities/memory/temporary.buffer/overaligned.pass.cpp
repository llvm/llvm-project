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
#include <limits>
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
    assert(reinterpret_cast<std::uintptr_t>(ip.first) % alignof(A) == 0);
    std::return_temporary_buffer(ip.first);

    // C++17 [depr.temporary.buffer]/4
    // Returns: If n <= 0 or if no storage could be obtained,
    // returns a pair P such that P.first is a null pointer value and P.second == 0;
    {
      std::pair<A*, std::ptrdiff_t> ret = std::get_temporary_buffer<A>(0);
      assert(ret.first == NULL);
      assert(ret.second == 0);
    }
    {
      TEST_DIAGNOSTIC_PUSH
      // This warning is coupled with completeness of control flow analysis which is affected by optimizations.
      TEST_GCC_DIAGNOSTIC_IGNORED("-Walloc-size-larger-than=")
      std::pair<A*, std::ptrdiff_t> ret = std::get_temporary_buffer<A>(std::numeric_limits<std::ptrdiff_t>::min());
      TEST_DIAGNOSTIC_POP
      assert(ret.first == NULL);
      assert(ret.second == 0);
    }

  return 0;
}
