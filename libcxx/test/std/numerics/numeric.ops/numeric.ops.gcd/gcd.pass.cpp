//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14

// <numeric>

// template<class _M, class _N>
// constexpr common_type_t<_M,_N> gcd(_M __m, _N __n)

#include <numeric>
#include <cassert>
#include <climits>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

#include "test_macros.h"

constexpr struct {
  int x;
  int y;
  int expect;
} Cases[] = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {2, 3, 1}, {2, 4, 2}, {11, 9, 1}, {36, 17, 1}, {36, 18, 18}};

template <typename Input1, typename Input2, typename Output>
constexpr bool test0(int in1, int in2, int out)
{
    auto value1 = static_cast<Input1>(in1);
    auto value2 = static_cast<Input2>(in2);
    static_assert(std::is_same_v<Output, decltype(std::gcd(value1, value2))>, "");
    static_assert(std::is_same_v<Output, decltype(std::gcd(value2, value1))>, "");
    assert(static_cast<Output>(out) == std::gcd(value1, value2));
    return true;
}

template <typename T>
T basic_gcd_(T m, T n) {
  return n == 0 ? m : basic_gcd_<T>(n, m % n);
}

template <typename T>
T basic_gcd(T m, T n) {
  using Tp = std::make_unsigned_t<T>;
  if constexpr (std::is_signed_v<T>) {
    if (m < 0 && m != std::numeric_limits<T>::min())
      m = -m;
    if (n < 0 && n != std::numeric_limits<T>::min())
      n = -n;
  }
  return basic_gcd_(static_cast<Tp>(m), static_cast<Tp>(n));
}

template <typename Input>
void do_fuzzy_tests() {
  std::mt19937 gen(1938);
  using DistIntType         = std::conditional_t<sizeof(Input) == 1, int, Input>; // See N4981 [rand.req.genl]/1.5
  constexpr Input max_input = std::numeric_limits<Input>::max();
  std::uniform_int_distribution<DistIntType> distrib(0, max_input);

  constexpr int nb_rounds = 10000;
  for (int i = 0; i < nb_rounds; ++i) {
    Input n = static_cast<Input>(distrib(gen));
    Input m = static_cast<Input>(distrib(gen));
    assert(std::gcd(n, m) == basic_gcd(n, m));
  }
}

template <typename Input>
void do_limit_tests() {
  Input inputs[] = {
      // The behavior of std::gcd is undefined if the absolute value of one of its
      // operand is not representable in the result type.
      std::numeric_limits<Input>::min() + (std::is_signed<Input>::value ? 3 : 0),
      std::numeric_limits<Input>::min() + 1,
      std::numeric_limits<Input>::min() + 2,
      std::numeric_limits<Input>::max(),
      std::numeric_limits<Input>::max() - 1,
      std::numeric_limits<Input>::max() - 2,
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      (Input)-1,
      (Input)-2,
      (Input)-3,
      (Input)-4,
      (Input)-5,
      (Input)-6,
      (Input)-7,
      (Input)-8,
      (Input)-9,
      (Input)-10,
  };

  for (auto n : inputs) {
    for (auto m : inputs) {
      assert(std::gcd(n, m) == basic_gcd(n, m));
    }
  }
}

template <typename Input1, typename Input2 = Input1>
constexpr bool do_test(int = 0)
{
    using S1 = std::make_signed_t<Input1>;
    using S2 = std::make_signed_t<Input2>;
    using U1 = std::make_unsigned_t<Input1>;
    using U2 = std::make_unsigned_t<Input2>;
    bool accumulate = true;
    for (auto TC : Cases) {
        { // Test with two signed types
            using Output = std::common_type_t<S1, S2>;
            accumulate &= test0<S1, S2, Output>(TC.x, TC.y, TC.expect);
            accumulate &= test0<S1, S2, Output>(-TC.x, TC.y, TC.expect);
            accumulate &= test0<S1, S2, Output>(TC.x, -TC.y, TC.expect);
            accumulate &= test0<S1, S2, Output>(-TC.x, -TC.y, TC.expect);
            accumulate &= test0<S2, S1, Output>(TC.x, TC.y, TC.expect);
            accumulate &= test0<S2, S1, Output>(-TC.x, TC.y, TC.expect);
            accumulate &= test0<S2, S1, Output>(TC.x, -TC.y, TC.expect);
            accumulate &= test0<S2, S1, Output>(-TC.x, -TC.y, TC.expect);
        }
        { // test with two unsigned types
            using Output = std::common_type_t<U1, U2>;
            accumulate &= test0<U1, U2, Output>(TC.x, TC.y, TC.expect);
            accumulate &= test0<U2, U1, Output>(TC.x, TC.y, TC.expect);
        }
        { // Test with mixed signs
            using Output = std::common_type_t<S1, U2>;
            accumulate &= test0<S1, U2, Output>(TC.x, TC.y, TC.expect);
            accumulate &= test0<U2, S1, Output>(TC.x, TC.y, TC.expect);
            accumulate &= test0<S1, U2, Output>(-TC.x, TC.y, TC.expect);
            accumulate &= test0<U2, S1, Output>(TC.x, -TC.y, TC.expect);
        }
        { // Test with mixed signs
            using Output = std::common_type_t<S2, U1>;
            accumulate &= test0<S2, U1, Output>(TC.x, TC.y, TC.expect);
            accumulate &= test0<U1, S2, Output>(TC.x, TC.y, TC.expect);
            accumulate &= test0<S2, U1, Output>(-TC.x, TC.y, TC.expect);
            accumulate &= test0<U1, S2, Output>(TC.x, -TC.y, TC.expect);
        }
    }
    return accumulate;
}

int main(int argc, char**)
{
    int non_cce = argc; // a value that can't possibly be constexpr

    static_assert(do_test<signed char>(), "");
    static_assert(do_test<short>(), "");
    static_assert(do_test<int>(), "");
    static_assert(do_test<long>(), "");
    static_assert(do_test<long long>(), "");

    assert(do_test<signed char>(non_cce));
    assert(do_test<short>(non_cce));
    assert(do_test<int>(non_cce));
    assert(do_test<long>(non_cce));
    assert(do_test<long long>(non_cce));

    static_assert(do_test<std::int8_t>(), "");
    static_assert(do_test<std::int16_t>(), "");
    static_assert(do_test<std::int32_t>(), "");
    static_assert(do_test<std::int64_t>(), "");

    assert(do_test<std::int8_t>(non_cce));
    assert(do_test<std::int16_t>(non_cce));
    assert(do_test<std::int32_t>(non_cce));
    assert(do_test<std::int64_t>(non_cce));

    static_assert(do_test<signed char, int>(), "");
    static_assert(do_test<int, signed char>(), "");
    static_assert(do_test<short, int>(), "");
    static_assert(do_test<int, short>(), "");
    static_assert(do_test<int, long>(), "");
    static_assert(do_test<long, int>(), "");
    static_assert(do_test<int, long long>(), "");
    static_assert(do_test<long long, int>(), "");

    assert((do_test<signed char, int>(non_cce)));
    assert((do_test<int, signed char>(non_cce)));
    assert((do_test<short, int>(non_cce)));
    assert((do_test<int, short>(non_cce)));
    assert((do_test<int, long>(non_cce)));
    assert((do_test<long, int>(non_cce)));
    assert((do_test<int, long long>(non_cce)));
    assert((do_test<long long, int>(non_cce)));

//  LWG#2837
    {
    auto res = std::gcd(static_cast<std::int64_t>(1234), INT32_MIN);
    static_assert(std::is_same_v<decltype(res), std::int64_t>, "");
    assert(res == 2);
    }

    do_fuzzy_tests<std::int8_t>();
    do_fuzzy_tests<std::int16_t>();
    do_fuzzy_tests<std::int32_t>();
    do_fuzzy_tests<std::int64_t>();
    do_fuzzy_tests<std::uint8_t>();
    do_fuzzy_tests<std::uint16_t>();
    do_fuzzy_tests<std::uint32_t>();
    do_fuzzy_tests<std::uint64_t>();

    do_limit_tests<std::int8_t>();
    do_limit_tests<std::int16_t>();
    do_limit_tests<std::int32_t>();
    do_limit_tests<std::int64_t>();
    do_limit_tests<std::uint8_t>();
    do_limit_tests<std::uint16_t>();
    do_limit_tests<std::uint32_t>();
    do_limit_tests<std::uint64_t>();

    return 0;
}
