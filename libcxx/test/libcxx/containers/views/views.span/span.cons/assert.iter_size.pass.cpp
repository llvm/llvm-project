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
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <array>
#include <span>

#include "check_assertion.h"

int main(int, char**) {
    std::array<int, 3> array{0, 1, 2};

    auto too_large = [&] { std::span<int, 3> const s(array.data(), 4); (void)s; };
    TEST_LIBCPP_ASSERT_FAILURE(too_large(), "size mismatch in span's constructor (iterator, len)");

    auto too_small = [&] { std::span<int, 3> const s(array.data(), 2); (void)s; };
    TEST_LIBCPP_ASSERT_FAILURE(too_small(), "size mismatch in span's constructor (iterator, len)");

    return 0;
}
