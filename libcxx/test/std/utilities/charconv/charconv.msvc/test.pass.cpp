//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// to_chars requires functions in the dylib that were introduced in Mac OS 10.15.
//
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{.+}}

// steady_clock requires threads.
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: libcpp-has-no-random-device
// UNSUPPORTED: libcpp-has-no-localization

// TODO(ldionne): This test fails on Ubuntu Focal on our CI nodes (and only there), in 32 bit mode.
// UNSUPPORTED: linux && 32bits-on-64bits

// XFAIL: LIBCXX-AIX-FIXME

// <charconv>

#include <type_traits>

// Work-around for sprintf_s's usage in the Microsoft tests.
#ifndef _WIN32
#  define sprintf_s snprintf
#endif

// FUNCTION TEMPLATE _Bit_cast
template <class _To, class _From,
          std::enable_if_t<sizeof(_To) == sizeof(_From) && std::is_trivially_copyable_v<_To> &&
                               std::is_trivially_copyable_v<_From>,
                           int> = 0>
[[nodiscard]] constexpr _To _Bit_cast(const _From& _From_obj) noexcept {
  return __builtin_bit_cast(_To, _From_obj);
}

// Includes Microsoft's test that tests the entire header.

#include "test.cpp"
