//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// MODULES_DEFINES: _LIBCPP_DEBUG=1
// MODULES_DEFINES: _LIBCPP_DEBUG_USE_EXCEPTIONS

// Can't test the system lib because this test enables debug mode
// UNSUPPORTED: with_system_cxx_lib

// test array<T, 0>::operator[] throws a debug exception.

#define _LIBCPP_DEBUG 1
#define _LIBCPP_DEBUG_USE_EXCEPTIONS
#include <array>

template <class Array>
inline bool CheckDebugThrows(Array& Arr, size_t Index) {
  try {
    Arr[Index];
  } catch (std::__libcpp_debug_exception const&) {
    return true;
  }
  return false;
}

int main()
{
  {
    typedef std::array<int, 0> C;
    C c = {};
    C const& cc = c;
    assert(CheckDebugThrows(c, 0));
    assert(CheckDebugThrows(c, 1));
    assert(CheckDebugThrows(cc, 0));
    assert(CheckDebugThrows(cc, 1));
  }
  {
    typedef std::array<const int, 0> C;
    C c = {{}};
    C const& cc = c;
    assert(CheckDebugThrows(c, 0));
    assert(CheckDebugThrows(c, 1));
    assert(CheckDebugThrows(cc, 0));
    assert(CheckDebugThrows(cc, 1));
  }
}
