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
// constexpr span<T, Extent>::span(const span<U, dynamic_extent>& other);
//
// Check that we ensure `other.size() == Extent`.

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-hardened-mode && !libcpp-has-debug-mode && !libcpp-has-assertions
// XFAIL: availability-verbose_abort-missing

#include <array>
#include <span>

#include "check_assertion.h"

int main(int, char**) {
    std::array<int, 3> array{0, 1, 2};
    std::span<int> other(array.data(), 3);

    auto invalid_source = [&] { std::span<int, 2> const s(other); (void)s; };
    TEST_LIBCPP_ASSERT_FAILURE(invalid_source(), "size mismatch in span's constructor (other span)");

    return 0;
}
