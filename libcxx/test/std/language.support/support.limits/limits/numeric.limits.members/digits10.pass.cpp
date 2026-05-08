//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// digits10

#include <limits>
#include <cfloat>

#include "test_macros.h"

template <class T, int expected>
void
test()
{
    static_assert(std::numeric_limits<T>::digits10 == expected, "digits10 test 1");
    static_assert(std::numeric_limits<T>::is_bounded, "digits10 test 5");
    static_assert(std::numeric_limits<const T>::digits10 == expected, "digits10 test 2");
    static_assert(std::numeric_limits<const T>::is_bounded, "digits10 test 6");
    static_assert(std::numeric_limits<volatile T>::digits10 == expected, "digits10 test 3");
    static_assert(std::numeric_limits<volatile T>::is_bounded, "digits10 test 7");
    static_assert(std::numeric_limits<const volatile T>::digits10 == expected, "digits10 test 4");
    static_assert(std::numeric_limits<const volatile T>::is_bounded, "digits10 test 8");
}

int main(int, char**)
{
    test<bool, 0>();
    test<char, 2>();
    test<signed char, 2>();
    test<unsigned char, 2>();
    test<wchar_t, 5*sizeof(wchar_t)/2-1>();  // 4 -> 9 and 2 -> 4
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t, 2>();
#endif
    test<char16_t, 4>();
    test<char32_t, 9>();
    test<short, 4>();
    test<unsigned short, 4>();
    test<int, 9>();
    test<unsigned int, 9>();
    test<long, sizeof(long) == 4 ? 9 : 18>();
    test<unsigned long, sizeof(long) == 4 ? 9 : 19>();
    test<long long, 18>();
    test<unsigned long long, 19>();
#ifndef TEST_HAS_NO_INT128
    test<__int128_t, 38>();
    test<__uint128_t, 38>();
#endif
    test<float, FLT_DIG>();
    test<double, DBL_DIG>();
    test<long double, LDBL_DIG>();

    // _BitInt(N): digits10 = floor((N - is_signed) * log10(2)).
#if TEST_HAS_EXTENSION(bit_int)
    test<unsigned _BitInt(8), 2>();   // digits=8,   log10=2.4
    test<signed _BitInt(8), 2>();     // digits=7,   log10=2.1
    test<unsigned _BitInt(13), 3>();  // digits=13,  log10=3.9
    test<signed _BitInt(13), 3>();    // digits=12,  log10=3.6
    test<unsigned _BitInt(32), 9>();  // digits=32,  log10=9.6
    test<unsigned _BitInt(37), 11>(); // digits=37,  log10=11.1
    test<unsigned _BitInt(64), 19>(); // digits=64,  log10=19.3
    test<signed _BitInt(64), 18>();   // digits=63,  log10=18.9
#  if __BITINT_MAXWIDTH__ >= 128
    test<unsigned _BitInt(77), 23>();  // digits=77,  log10=23.2
    test<signed _BitInt(77), 22>();    // digits=76,  log10=22.9
    test<unsigned _BitInt(128), 38>(); // digits=128, log10=38.5
    test<signed _BitInt(128), 38>();   // digits=127, log10=38.2
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
    test<unsigned _BitInt(129), 38>(); // digits=129, log10=38.8
    test<unsigned _BitInt(255), 76>(); // digits=255, log10=76.8
    test<unsigned _BitInt(256), 77>(); // digits=256, log10=77.1
    test<signed _BitInt(256), 76>();   // digits=255, log10=76.8
    test<unsigned _BitInt(257), 77>(); // digits=257, log10=77.4
#  endif
#  if __BITINT_MAXWIDTH__ >= 4096
    test<unsigned _BitInt(4096), 1233>(); // digits=4096, log10=1233.0
    test<signed _BitInt(4096), 1232>();   // digits=4095, log10=1232.7
#  endif

    // Very wide _BitInt: pin the log10(2) approximation used by digits10.
    // Each width is the first point at which a coarser rational convergent
    // of log10(2) would give the wrong floor, so these tests bite if the
    // formula ever regresses.
#  if __BITINT_MAXWIDTH__ >= 15437
    // A coarser convergent (643/2136) would give 4646 here; correct is 4647.
    test<unsigned _BitInt(15437), 4647>();
#  endif
#  if __BITINT_MAXWIDTH__ >= 70777
    // A coarser convergent (8651/28738) would give 21305 here; correct is 21306.
    test<unsigned _BitInt(70777), 21306>();
#  endif
#  if __BITINT_MAXWIDTH__ >= 1000000
    test<unsigned _BitInt(1000000), 301029>();
#  endif
#  if __BITINT_MAXWIDTH__ >= 8388608
    // Pin the exact upper bound of the approximation.
    test<unsigned _BitInt(8388608), 2525222>();
#  endif
    // The 1936274/6432163 convergent stays exact up to d=51132156. 8388608 is
    // the largest width tested above, so if Clang raises __BITINT_MAXWIDTH__,
    // extend the coverage before trusting the formula at the new range.
    LIBCPP_STATIC_ASSERT(__BITINT_MAXWIDTH__ <= 8388608);
#endif // TEST_HAS_EXTENSION(bit_int)

    return 0;
}
