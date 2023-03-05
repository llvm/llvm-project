//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <stdlib.h>

#include <stdlib.h>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

#ifdef abs
#error abs is defined
#endif

#ifdef labs
#error labs is defined
#endif

#ifdef llabs
#error llabs is defined
#endif

#ifdef div
#error div is defined
#endif

#ifdef ldiv
#error ldiv is defined
#endif

#ifdef lldiv
#error lldiv is defined
#endif

#ifndef EXIT_FAILURE
#error EXIT_FAILURE not defined
#endif

#ifndef EXIT_SUCCESS
#error EXIT_SUCCESS not defined
#endif

#ifndef MB_CUR_MAX
#error MB_CUR_MAX not defined
#endif

#ifndef NULL
#error NULL not defined
#endif

#ifndef RAND_MAX
#error RAND_MAX not defined
#endif

template <class T, class = decltype(::abs(std::declval<T>()))>
std::true_type has_abs_imp(int);
template <class T>
std::false_type has_abs_imp(...);

template <class T>
struct has_abs : decltype(has_abs_imp<T>(0)) {};

void test_abs() {
  TEST_DIAGNOSTIC_PUSH
  TEST_CLANG_DIAGNOSTIC_IGNORED("-Wabsolute-value")
  ASSERT_SAME_TYPE(float,       decltype(abs((float)0)));
  ASSERT_SAME_TYPE(double,      decltype(abs((double)0)));
  ASSERT_SAME_TYPE(long double, decltype(abs((long double)0)));
  ASSERT_SAME_TYPE(int,         decltype(abs((int)0)));
  ASSERT_SAME_TYPE(long,        decltype(abs((long)0)));
  ASSERT_SAME_TYPE(long long,   decltype(abs((long long)0)));
  ASSERT_SAME_TYPE(int,         decltype(abs((unsigned char)0)));
  ASSERT_SAME_TYPE(int,         decltype(abs((unsigned short)0)));
  ASSERT_SAME_TYPE(int,         decltype(abs((signed char)0)));
  ASSERT_SAME_TYPE(int,         decltype(abs((short)0)));
  ASSERT_SAME_TYPE(int,         decltype(abs((unsigned char)0)));
  ASSERT_SAME_TYPE(int,         decltype(abs((char)0)));

  static_assert(!has_abs<unsigned>::value, "");
  static_assert(!has_abs<unsigned long>::value, "");
  static_assert(!has_abs<unsigned long long>::value, "");
  static_assert(!has_abs<size_t>::value, "");

  TEST_DIAGNOSTIC_POP

  assert(abs(-1.) == 1);
}

int main(int, char**) {
    size_t s = 0; ((void)s);
    div_t d; ((void)d);
    ldiv_t ld; ((void)ld);
    lldiv_t lld; ((void)lld);
    char** endptr = 0;
    ASSERT_SAME_TYPE(double,             decltype(atof("")));
    ASSERT_SAME_TYPE(int,                decltype(atoi("")));
    ASSERT_SAME_TYPE(long,               decltype(atol("")));
    ASSERT_SAME_TYPE(long long,          decltype(atoll("")));
    ASSERT_SAME_TYPE(char*,              decltype(getenv("")));
    ASSERT_SAME_TYPE(double,             decltype(strtod("", endptr)));
    ASSERT_SAME_TYPE(float,              decltype(strtof("", endptr)));
    ASSERT_SAME_TYPE(long double,        decltype(strtold("", endptr)));
    ASSERT_SAME_TYPE(long,               decltype(strtol("", endptr,0)));
    ASSERT_SAME_TYPE(long long,          decltype(strtoll("", endptr,0)));
    ASSERT_SAME_TYPE(unsigned long,      decltype(strtoul("", endptr,0)));
    ASSERT_SAME_TYPE(unsigned long long, decltype(strtoull("", endptr,0)));
    ASSERT_SAME_TYPE(int,                decltype(rand()));
    ASSERT_SAME_TYPE(void,               decltype(srand(0)));

    // aligned_alloc tested in stdlib_h.aligned_alloc.compile.pass.cpp

    void* pv = 0;
    void (*handler)() = 0;
    int (*comp)(void const*, void const*) = 0;
    ASSERT_SAME_TYPE(void*,     decltype(calloc(0,0)));
    ASSERT_SAME_TYPE(void,      decltype(free(0)));
    ASSERT_SAME_TYPE(void*,     decltype(malloc(0)));
    ASSERT_SAME_TYPE(void*,     decltype(realloc(0,0)));
    ASSERT_SAME_TYPE(void,      decltype(abort()));
    ASSERT_SAME_TYPE(int,       decltype(atexit(handler)));
    ASSERT_SAME_TYPE(void,      decltype(exit(0)));
    ASSERT_SAME_TYPE(void,      decltype(_Exit(0)));
    ASSERT_SAME_TYPE(char*,     decltype(getenv("")));
    ASSERT_SAME_TYPE(int,       decltype(system("")));
    ASSERT_SAME_TYPE(void*,     decltype(bsearch(pv,pv,0,0,comp)));
    ASSERT_SAME_TYPE(void,      decltype(qsort(pv,0,0,comp)));
    ASSERT_SAME_TYPE(int,       decltype(abs(0)));
    ASSERT_SAME_TYPE(long,      decltype(labs((long)0)));
    ASSERT_SAME_TYPE(long long, decltype(llabs((long long)0)));
    ASSERT_SAME_TYPE(div_t,     decltype(div(0,0)));
    ASSERT_SAME_TYPE(ldiv_t,    decltype(ldiv(0L,0L)));
    ASSERT_SAME_TYPE(lldiv_t,   decltype(lldiv(0LL,0LL)));
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    wchar_t* pw = 0;
    const wchar_t* pwc = 0;
    char* pc = 0;
    ASSERT_SAME_TYPE(int,    decltype(mblen("",0)));
    ASSERT_SAME_TYPE(int,    decltype(mbtowc(pw,"",0)));
    ASSERT_SAME_TYPE(int,    decltype(wctomb(pc,L' ')));
    ASSERT_SAME_TYPE(size_t, decltype(mbstowcs(pw,"",0)));
    ASSERT_SAME_TYPE(size_t, decltype(wcstombs(pc,pwc,0)));
#endif

    test_abs();

    return 0;
}
