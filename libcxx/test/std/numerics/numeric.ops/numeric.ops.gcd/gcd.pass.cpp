//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++98, c++03, c++11, c++14
// <numeric>

// template<class _M, class _N>
// constexpr common_type_t<_M,_N> gcd(_M __m, _N __n)

#include <numeric>
#include <cassert>
#include <cstdlib>    // for rand()
#include <iostream>

constexpr struct {
  int x;
  int y;
  int expect;
} Cases[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
    {2, 3, 1},
    {2, 4, 2},
    {36, 17, 1},
    {36, 18, 18}
};


template <typename Input1, typename Input2, typename Output>
constexpr bool test0(Input1 in1, Input2 in2, Output out)
{
    static_assert((std::is_same<Output, decltype(std::gcd(in1, in2))>::value), "" );
    static_assert((std::is_same<Output, decltype(std::gcd(in2, in1))>::value), "" );
    return out == std::gcd(in1, in2) ? true : (std::abort(), false);
}


template <typename Input1, typename Input2 = Input1>
constexpr bool do_test(int dummy = 0)
{
    using S1 = typename std::make_signed<Input1>::type;
    using S2 = typename std::make_signed<Input2>::type;
    using U1 = typename std::make_unsigned<Input1>::type;
    using U2 = typename std::make_unsigned<Input2>::type;
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

int main()
{
    auto non_cce = std::rand(); // a value that can't possibly be constexpr

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

    static_assert(do_test< int8_t>(), "");
    static_assert(do_test<int16_t>(), "");
    static_assert(do_test<int32_t>(), "");
    static_assert(do_test<int64_t>(), "");

    assert(do_test< int8_t>(non_cce));
    assert(do_test<int16_t>(non_cce));
    assert(do_test<int32_t>(non_cce));
    assert(do_test<int64_t>(non_cce));

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
}
