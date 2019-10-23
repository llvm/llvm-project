//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "test_macros.h"

template<class T>
struct correct_size_int
{
    typedef typename std::conditional<sizeof(T) < sizeof(int), int, T>::type type;
};

template <class Source, class Result>
void test_abs()
{
    Source neg_val = -5;
    Source pos_val = 5;
    Result res = 5;

    ASSERT_SAME_TYPE(decltype(std::abs(neg_val)), Result);

    assert(std::abs(neg_val) == res);
    assert(std::abs(pos_val) == res);
}

void test_big()
{
    long long int big_value = std::numeric_limits<long long int>::max(); // a value too big for ints to store
    long long int negative_big_value = -big_value;
    assert(std::abs(negative_big_value) == big_value); // make sure it doesn't get casted to a smaller type
}

// The following is helpful to keep in mind:
// 1byte == char <= short <= int <= long <= long long

int main(int, char**)
{
    // On some systems char is unsigned.
    // If that is the case, we should just test signed char twice.
    typedef typename std::conditional<
        std::is_signed<char>::value, char, signed char
    >::type SignedChar;

    // All types less than or equal to and not greater than int are promoted to int.
    test_abs<short int, int>();
    test_abs<SignedChar, int>();
    test_abs<signed char, int>();

    // These three calls have specific overloads:
    test_abs<int, int>();
    test_abs<long int, long int>();
    test_abs<long long int, long long int>();

    // Here there is no guarantee that int is larger than int8_t so we
    // use a helper type trait to conditional test against int.
    test_abs<std::int8_t, typename correct_size_int<std::int8_t>::type>();
    test_abs<std::int16_t, typename correct_size_int<std::int16_t>::type>();
    test_abs<std::int32_t, typename correct_size_int<std::int32_t>::type>();
    test_abs<std::int64_t, typename correct_size_int<std::int64_t>::type>();

    test_abs<long double, long double>();
    test_abs<double, double>();
    test_abs<float, float>();

    test_big();

    return 0;
}
