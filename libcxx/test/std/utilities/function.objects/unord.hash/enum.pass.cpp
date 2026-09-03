//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// Make sure that we can hash enumeration values.

#include <functional>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "constexpr_hash.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11
#  include "poisoned_hash_helper.h"
#endif

enum class Colors { red, orange, yellow, green, blue, indigo, violet };
enum class Cardinals { zero, one, two, three, five=5 };
enum class LongColors : short { red, orange, yellow, green, blue, indigo, violet };
enum class ShortColors : long { red, orange, yellow, green, blue, indigo, violet };
enum class EightBitColors : std::uint8_t { red, orange, yellow, green, blue, indigo, violet };

enum Fruits { apple, pear, grape, mango, cantaloupe };

template <class T, template <class> typename THash>
TEST_CONSTEXPR_CXX26 void test() {
#if TEST_STD_VER >= 11
    test_hash_disabled<const T>();
    test_hash_disabled<volatile T>();
    test_hash_disabled<const volatile T>();
#endif

    // typedef std::hash<T> H;
    typedef THash<T> H;
#if TEST_STD_VER <= 17
    static_assert((std::is_same<typename H::argument_type, T>::value), "");
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "");
#endif
    ASSERT_NOEXCEPT(H()(T()));
    typedef typename std::underlying_type<T>::type under_type;

    H h1;
    // std::hash<under_type> h2;
    THash<under_type> h2;
    for (int i = 0; i <= 5; ++i)
    {
        T t(static_cast<T> (i));
        const bool small = std::integral_constant<bool, sizeof(T) <= sizeof(std::size_t)>::value; // avoid compiler warnings
        if (small)
            assert(h1(t) == h2(static_cast<under_type>(i)));
    }
}

template <template <typename> typename THash >
TEST_CONSTEXPR_CXX26 bool test_with_hash() {
  test<Cardinals, THash>();

  test<Colors, THash>();
  test<ShortColors, THash>();
  test<LongColors, THash>();
  test<EightBitColors, THash>();

  test<Fruits, THash>();

  return true;
}

int main(int, char**) {
  assert(test_with_hash<std::hash>());

#if TEST_STD_VER >= 26
  static_assert(test_with_hash<support::constexpr_hash>());
#endif

  return 0;
}
