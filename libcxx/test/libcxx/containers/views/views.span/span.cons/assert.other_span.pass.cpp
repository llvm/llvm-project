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
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

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
