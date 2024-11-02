//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class UIntType, size_t w, size_t s, size_t r>
// class subtract_with_carry_engine;

// explicit subtract_with_carry_engine(result_type s = default_seed);         // before C++20
// subtract_with_carry_engine() : subtract_with_carry_engine(default_seed) {} // C++20
// explicit subtract_with_carry_engine(result_type s);                        // C++20

// Serializing/deserializing the state of the RNG requires iostreams
// UNSUPPORTED: no-localization

#include <random>
#include <sstream>
#include <cassert>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "make_implicit.h"
#include "test_convertible.h"
#endif

template <class T>
std::string
to_string(T const &e)
{
    std::ostringstream os;
    os << e;
    return os.str();
}

void
test1()
{
    const char* a = "15136306 8587749 2346244 16479026 15515802 9510553 "
    "16090340 14501685 13839944 10789678 11581259 9590790 5840316 5953700 "
    "13398366 8134459 16629731 6851902 15583892 1317475 4231148 9092691 "
    "5707268 2355175 0";
    std::ranlux24_base e1(0);
    assert(to_string(e1) == a);
}

void
test2()
{
    const char* a = "10880375256626 126660097854724 33643165434010 "
    "78293780235492 179418984296008 96783156950859 238199764491708 "
    "34339434557790 155299155394531 29014415493780 209265474179052 "
    "263777435457028 0";
    std::ranlux48_base e1(0);
    assert(to_string(e1) == a);
}

#if TEST_STD_VER >= 11
template <class E>
void test_implicit_ctor() {
  assert(E(E::default_seed) == make_implicit<E>());
}
#endif

int main(int, char**)
{
    test1();
    test2();

#if TEST_STD_VER >= 11
    static_assert(test_convertible<std::ranlux24_base>(), "");
    static_assert(test_convertible<std::ranlux48_base>(), "");
    test_implicit_ctor<std::ranlux24_base>();
    test_implicit_ctor<std::ranlux48_base>();
    static_assert(!test_convertible<std::ranlux24_base, uint_fast32_t>(), "");
    static_assert(!test_convertible<std::ranlux48_base, uint_fast64_t>(), "");
#endif

  return 0;
}
