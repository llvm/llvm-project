//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include "test_macros.h"

TEST_CLANG_DIAGNOSTIC_IGNORED("-Wsign-compare")
TEST_GCC_DIAGNOSTIC_IGNORED("-Wsign-compare")
TEST_MSVC_DIAGNOSTIC_IGNORED(4018 4389) // various "signed/unsigned mismatch"

// constexpr iota_view(type_identity_t<W> value, type_identity_t<Bound> bound); // explicit since C++23

#include <ranges>
#include <cassert>

#include "test_convertible.h"
#include "types.h"

// SFINAE tests.

#if TEST_STD_VER >= 23

static_assert(!test_convertible<std::ranges::iota_view<SomeInt, SomeInt>,
                                decltype(std::ranges::iota_view<SomeInt, SomeInt>{}.begin()),
                                decltype(std::ranges::iota_view<SomeInt, SomeInt>{}.end())>(),
              "This constructor must be explicit");

static_assert(!test_convertible<std::ranges::iota_view<SomeInt>,
                                decltype(std::ranges::iota_view<SomeInt>{}.begin()),
                                decltype(std::unreachable_sentinel)>(),
              "This constructor must be explicit");

static_assert(!test_convertible<std::ranges::iota_view<SomeInt, IntComparableWith<SomeInt>>,
                                decltype(std::ranges::iota_view{SomeInt(0), IntComparableWith(SomeInt(10))}.begin()),
                                decltype(std::ranges::iota_view{SomeInt(0), IntComparableWith(SomeInt(10))}.end())>(),
              "This constructor must be explicit");

#else

static_assert( test_convertible<std::ranges::iota_view<SomeInt, SomeInt>,
                                decltype(std::ranges::iota_view<SomeInt, SomeInt>{}.begin()),
                                decltype(std::ranges::iota_view<SomeInt, SomeInt>{}.end())>(),
              "This constructor must not be explicit");

static_assert( test_convertible<std::ranges::iota_view<SomeInt>,
                                decltype(std::ranges::iota_view<SomeInt>{}.begin()),
                                decltype(std::unreachable_sentinel)>(),
              "This constructor must not be explicit");

static_assert( test_convertible<std::ranges::iota_view<SomeInt, IntComparableWith<SomeInt>>,
                                decltype(std::ranges::iota_view{SomeInt(0), IntComparableWith(SomeInt(10))}.begin()),
                                decltype(std::ranges::iota_view{SomeInt(0), IntComparableWith(SomeInt(10))}.end())>(),
              "This constructor must not be explicit");

#endif // TEST_STD_VER >= 23

constexpr bool test() {
  {
    std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(0), SomeInt(10));
    assert(std::ranges::next(io.begin(), 10) == io.end());
  }

  {
    std::ranges::iota_view<SomeInt> io(SomeInt(0), std::unreachable_sentinel);
    assert(std::ranges::next(io.begin(), 10) != io.end());
  }

  {
    std::ranges::iota_view<SomeInt, IntComparableWith<SomeInt>> io(SomeInt(0), IntComparableWith(SomeInt(10)));
    assert(std::ranges::next(io.begin(), 10) == io.end());
  }

  {
    // This is allowed only when using the constructor (not the deduction guide).
    std::ranges::iota_view<int, unsigned> signedUnsigned(0, 10);
    assert(std::ranges::next(signedUnsigned.begin(), 10) == signedUnsigned.end());
  }

  {
    // This is allowed only when using the constructor (not the deduction guide).
    std::ranges::iota_view<unsigned, int> signedUnsigned(0, 10);
    assert(std::ranges::next(signedUnsigned.begin(), 10) == signedUnsigned.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
