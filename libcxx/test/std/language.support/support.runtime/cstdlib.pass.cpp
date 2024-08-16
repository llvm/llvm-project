//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <cstdlib>

#include <cstdlib>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

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

template <class T, class = decltype(std::abs(std::declval<T>()))>
std::true_type has_abs_imp(int);
template <class T>
std::false_type has_abs_imp(...);

template <class T>
struct has_abs : decltype(has_abs_imp<T>(0)) {};

void test_abs() {
    TEST_DIAGNOSTIC_PUSH
    TEST_CLANG_DIAGNOSTIC_IGNORED("-Wabsolute-value")
    static_assert((std::is_same<decltype(std::abs((float)0)), float>::value), "");
    static_assert((std::is_same<decltype(std::abs((double)0)), double>::value), "");
    static_assert(
        (std::is_same<decltype(std::abs((long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::abs((int)0)), int>::value), "");
    static_assert((std::is_same<decltype(std::abs((long)0)), long>::value), "");
    static_assert((std::is_same<decltype(std::abs((long long)0)), long long>::value),
                  "");
    static_assert((std::is_same<decltype(std::abs((unsigned char)0)), int>::value),
                  "");
    static_assert((std::is_same<decltype(std::abs((unsigned short)0)), int>::value),
                  "");
    static_assert((std::is_same<decltype(std::abs((signed char)0)), int>::value),
                  "");
    static_assert((std::is_same<decltype(std::abs((short)0)), int>::value),
                  "");
    static_assert((std::is_same<decltype(std::abs((unsigned char)0)), int>::value),
                  "");
    static_assert((std::is_same<decltype(std::abs((char)0)), int>::value),
                  "");

    static_assert(!has_abs<unsigned>::value, "");
    static_assert(!has_abs<unsigned long>::value, "");
    static_assert(!has_abs<unsigned long long>::value, "");
    static_assert(!has_abs<std::size_t>::value, "");

    TEST_DIAGNOSTIC_POP

    assert(std::abs(-1.) == 1);
}

template <class TestType, class IntType>
void test_div_struct() {
  TestType obj;
  static_assert(sizeof(obj) >= sizeof(IntType) * 2,
                "");  // >= to account for alignment.
  static_assert((std::is_same<decltype(obj.quot), IntType>::value), "");
  static_assert((std::is_same<decltype(obj.rem), IntType>::value), "");
  ((void)obj);
}

void test_div() {
  {  // tests member types of std::div_t, etc.
    test_div_struct<std::div_t, int>();
    test_div_struct<std::ldiv_t, long>();
    test_div_struct<std::lldiv_t, long long>();
  }

  {  // tests return type of std::div
    // clang-format off
    static_assert((std::is_same<decltype(std::div(  0,   0  )), std::div_t  >::value), "");
    static_assert((std::is_same<decltype(std::div(  0L,  0L )), std::ldiv_t >::value), "");
    static_assert((std::is_same<decltype(std::div(  0LL, 0LL)), std::lldiv_t>::value), "");
    static_assert((std::is_same<decltype(std::ldiv( 0L,  0L )), std::ldiv_t >::value), "");
    static_assert((std::is_same<decltype(std::lldiv(0LL, 0LL)), std::lldiv_t>::value), "");
    // clang-format on
  }

  {  // check one basic input for correctness.
    // (42 // 5 == 8) AND (42 % 5 == 2)
    const auto check = [](const auto callable_div) -> void {
      const auto div = callable_div(42, 5);
      assert(div.quot == 8);
      assert(div.rem == 2);

#if _LIBCPP_STD_VER >= 23
      constexpr auto div2 = callable_div(42, 5);
      static_assert(div2.quot == 8);
      static_assert(div2.rem == 2);
#endif
    };

    // clang-format off
    check([](int       n, int       k) { return std::div(  n, k); });
    check([](long      n, long      k) { return std::div(  n, k); });
    check([](long long n, long long k) { return std::div(  n, k); });
    // check([](long      n, long      k) { return std::ldiv( n, k); });
    // check([](long long n, long long k) { return std::lldiv(n, k); });
    // clang-format on
  }
}

int main(int, char**)
{
    std::size_t s = 0;
    ((void)s);
    static_assert((std::is_same<std::size_t, decltype(sizeof(int))>::value), "");
    char** endptr = 0;
    static_assert((std::is_same<decltype(std::atof("")), double>::value), "");
    static_assert((std::is_same<decltype(std::atoi("")), int>::value), "");
    static_assert((std::is_same<decltype(std::atol("")), long>::value), "");
    static_assert((std::is_same<decltype(std::atoll("")), long long>::value), "");
    static_assert((std::is_same<decltype(std::getenv("")), char*>::value), "");
    static_assert((std::is_same<decltype(std::strtod("", endptr)), double>::value), "");
    static_assert((std::is_same<decltype(std::strtof("", endptr)), float>::value), "");
    static_assert((std::is_same<decltype(std::strtold("", endptr)), long double>::value), "");
    static_assert((std::is_same<decltype(std::strtol("", endptr,0)), long>::value), "");
    static_assert((std::is_same<decltype(std::strtoll("", endptr,0)), long long>::value), "");
    static_assert((std::is_same<decltype(std::strtoul("", endptr,0)), unsigned long>::value), "");
    static_assert((std::is_same<decltype(std::strtoull("", endptr,0)), unsigned long long>::value), "");
    static_assert((std::is_same<decltype(std::rand()), int>::value), "");
    static_assert((std::is_same<decltype(std::srand(0)), void>::value), "");

    // std::aligned_alloc tested in cstdlib.aligned_alloc.compile.pass.cpp

    void* pv = 0;
    void (*handler)() = 0;
    int (*comp)(void const*, void const*) = 0;
    static_assert((std::is_same<decltype(std::calloc(0,0)), void*>::value), "");
    static_assert((std::is_same<decltype(std::free(0)), void>::value), "");
    static_assert((std::is_same<decltype(std::malloc(0)), void*>::value), "");
    static_assert((std::is_same<decltype(std::realloc(0,0)), void*>::value), "");
    static_assert((std::is_same<decltype(std::abort()), void>::value), "");
    static_assert((std::is_same<decltype(std::atexit(handler)), int>::value), "");
    static_assert((std::is_same<decltype(std::exit(0)), void>::value), "");
    static_assert((std::is_same<decltype(std::_Exit(0)), void>::value), "");
    static_assert((std::is_same<decltype(std::getenv("")), char*>::value), "");
    static_assert((std::is_same<decltype(std::system("")), int>::value), "");
    static_assert((std::is_same<decltype(std::bsearch(pv,pv,0,0,comp)), void*>::value), "");
    static_assert((std::is_same<decltype(std::qsort(pv,0,0,comp)), void>::value), "");
    static_assert((std::is_same<decltype(std::abs(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::abs((long)0)), long>::value), "");
    static_assert((std::is_same<decltype(std::abs((long long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(std::labs((long)0)), long>::value), "");
    static_assert((std::is_same<decltype(std::llabs((long long)0)), long long>::value), "");
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    wchar_t* pw = 0;
    const wchar_t* pwc = 0;
    char* pc = 0;
    static_assert((std::is_same<decltype(std::mblen("",0)), int>::value), "");
    static_assert((std::is_same<decltype(std::mbtowc(pw,"",0)), int>::value), "");
    static_assert((std::is_same<decltype(std::wctomb(pc,L' ')), int>::value), "");
    static_assert((std::is_same<decltype(std::mbstowcs(pw,"",0)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::wcstombs(pc,pwc,0)), std::size_t>::value), "");
#endif

    test_abs();
    test_div();
}
