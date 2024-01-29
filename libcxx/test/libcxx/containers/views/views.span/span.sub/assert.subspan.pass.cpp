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
// template<size_t Offset, size_t Count = dynamic_extent>
// constexpr span<element_type, see-below> subspan() const;
//
// Requires: Offset <= size() && (Count == dynamic_extent || Count <= size() - Offset)
//
// constexpr span<element_type, dynamic_extent> subspan(
//   size_type offset, size_type count = dynamic_extent) const;
//
// Requires: offset <= size() && (count == dynamic_extent || count <= size() - offset)

// Make sure that creating a sub-span with an incorrect number of elements triggers an assertion.

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: availability-verbose_abort-missing

#include <array>
#include <span>
#include <cstddef>

#include "check_assertion.h"

int main(int, char**) {
    {
        std::array<int, 3> array{0, 1, 2};
        std::span<int> const s(array.data(), array.size());
        TEST_LIBCPP_ASSERT_FAILURE(s.subspan(4), "span<T>::subspan(offset, count): offset out of range");
        TEST_LIBCPP_ASSERT_FAILURE(s.subspan<4>(), "span<T>::subspan<Offset, Count>(): Offset out of range");

        TEST_LIBCPP_ASSERT_FAILURE(s.subspan(0, 4), "span<T>::subspan(offset, count): offset + count out of range");
        TEST_LIBCPP_ASSERT_FAILURE((s.subspan<0, 4>()), "span<T>::subspan<Offset, Count>(): Offset + Count out of range");

        TEST_LIBCPP_ASSERT_FAILURE(s.subspan(1, 3), "span<T>::subspan(offset, count): offset + count out of range");
        TEST_LIBCPP_ASSERT_FAILURE((s.subspan<1, 3>()), "span<T>::subspan<Offset, Count>(): Offset + Count out of range");
    }
    {
        std::array<int, 3> array{0, 1, 2};
        std::span<int, 3> const s(array.data(), array.size());
        TEST_LIBCPP_ASSERT_FAILURE(s.subspan(4), "span<T, N>::subspan(offset, count): offset out of range");
        // s.subspan<4>() caught at compile-time (tested in libcxx/test/std/containers/views/views.span/span.sub/subspan.verify.cpp)

        TEST_LIBCPP_ASSERT_FAILURE(s.subspan(0, 4), "span<T, N>::subspan(offset, count): offset + count out of range");
        // s.subspan<0, 4>() caught at compile-time (tested in libcxx/test/std/containers/views/views.span/span.sub/subspan.verify.cpp)

        TEST_LIBCPP_ASSERT_FAILURE(s.subspan(1, 3), "span<T, N>::subspan(offset, count): offset + count out of range");
        // s.subspan<1, 3>() caught at compile-time (tested in libcxx/test/std/containers/views/views.span/span.sub/subspan.verify.cpp)
    }

    return 0;
}
