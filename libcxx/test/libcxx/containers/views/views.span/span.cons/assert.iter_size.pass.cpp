//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>
//
// constexpr span<T, Extent>::span(Iterator, size_type);
//
// Check that the passed size is equal to the statically known extent.
// Note that it doesn't make sense to validate the incoming size in the
// dynamic_extent version.

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <array>
#include <span>

#include "check_assertion.h"

int main(int, char**) {
  std::array<int, 3> array{0, 1, 2};

  // Input range too large (exceeds the span extent)
  {
    auto f = [&] {
      std::span<int, 3> const s(array.data(), 4);
      (void)s;
    };
    TEST_LIBCPP_ASSERT_FAILURE(f(), "size mismatch in span's constructor (iterator, len)");
  }

  // Input range too small (doesn't fill the span)
  {
    auto f = [&] {
      std::span<int, 3> const s(array.data(), 2);
      (void)s;
    };
    TEST_LIBCPP_ASSERT_FAILURE(f(), "size mismatch in span's constructor (iterator, len)");
  }

  // Input range is non-empty but starts with a null pointer
  {
    // static extent
    {
      auto f = [&] {
        int* p = nullptr;
        std::span<int, 3> const s(p, 3);
        (void)s;
      };
      TEST_LIBCPP_ASSERT_FAILURE(f(), "passed nullptr with non-zero length in span's constructor (iterator, len)");
    }

    // dynamic extent
    {
      auto f = [&] {
        int* p = nullptr;
        std::span<int, std::dynamic_extent> const s(p, 1);
        (void)s;
      };
      TEST_LIBCPP_ASSERT_FAILURE(f(), "passed nullptr with non-zero length in span's constructor (iterator, len)");
    }
  }

  return 0;
}
