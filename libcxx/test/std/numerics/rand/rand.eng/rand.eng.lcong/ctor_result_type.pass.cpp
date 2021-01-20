//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template <class UIntType, UIntType a, UIntType c, UIntType m>
//   class linear_congruential_engine;

// explicit linear_congruential_engine(result_type s = default_seed);         // before C++20
// linear_congruential_engine() : linear_congruential_engine(default_seed) {} // C++20
// explicit linear_congruential_engine(result_type s);                        // C++20

// Serializing/deserializing the state of the RNG requires iostreams
// UNSUPPORTED: libcpp-has-no-localization

#include <random>
#include <sstream>
#include <cassert>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "make_implicit.h"
#include "test_convertible.h"
#endif

template <class T>
std::string to_string(T const& e) {
  std::ostringstream os;
  os << e;
  return os.str();
}

template <class T>
void
test1()
{
    // c % m != 0 && s % m != 0
    {
        typedef std::linear_congruential_engine<T, 2, 3, 7> E;
        E e(5);
        assert(to_string(e) == "5");
    }
    {
        typedef std::linear_congruential_engine<T, 2, 3, 0> E;
        E e(5);
        assert(to_string(e) == "5");
    }
    {
        typedef std::linear_congruential_engine<T, 2, 3, 4> E;
        E e(5);
        assert(to_string(e) == "1");
    }
}

template <class T>
void
test2()
{
    // c % m != 0 && s % m == 0
    {
        typedef std::linear_congruential_engine<T, 2, 3, 7> E;
        E e(7);
        assert(to_string(e) == "0");
    }
    {
        typedef std::linear_congruential_engine<T, 2, 3, 0> E;
        E e(0);
        assert(to_string(e) == "0");
    }
    {
        typedef std::linear_congruential_engine<T, 2, 3, 4> E;
        E e(4);
        assert(to_string(e) == "0");
    }
}

template <class T>
void
test3()
{
    // c % m == 0 && s % m != 0
    {
        typedef std::linear_congruential_engine<T, 2, 0, 7> E;
        E e(3);
        assert(to_string(e) == "3");
    }
    {
        typedef std::linear_congruential_engine<T, 2, 0, 0> E;
        E e(5);
        assert(to_string(e) == "5");
    }
    {
        typedef std::linear_congruential_engine<T, 2, 0, 4> E;
        E e(7);
        assert(to_string(e) == "3");
    }
}

template <class T>
void
test4()
{
    // c % m == 0 && s % m == 0
    {
        typedef std::linear_congruential_engine<T, 2, 0, 7> E;
        E e(7);
        assert(to_string(e) == "1");
    }
    {
        typedef std::linear_congruential_engine<T, 2, 0, 0> E;
        E e(0);
        assert(to_string(e) == "1");
    }
    {
        typedef std::linear_congruential_engine<T, 2, 0, 4> E;
        E e(8);
        assert(to_string(e) == "1");
    }
}

template <class T>
void test_implicit() {
#if TEST_STD_VER >= 11
  typedef std::linear_congruential_engine<T, 2, 0, 7> E;
  static_assert(test_convertible<E>(), "");
  assert(E(E::default_seed) == make_implicit<E>());
  static_assert(!test_convertible<E, T>(), "");
#endif
}

int main(int, char**)
{
    test1<unsigned short>();
    test1<unsigned int>();
    test1<unsigned long>();
    test1<unsigned long long>();

    test2<unsigned short>();
    test2<unsigned int>();
    test2<unsigned long>();
    test2<unsigned long long>();

    test3<unsigned short>();
    test3<unsigned int>();
    test3<unsigned long>();
    test3<unsigned long long>();

    test4<unsigned short>();
    test4<unsigned int>();
    test4<unsigned long>();
    test4<unsigned long long>();

    test_implicit<unsigned short>();

    return 0;
}
