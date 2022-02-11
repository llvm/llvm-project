//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class Iter, IntegralLike Size, class T>
//   requires OutputIterator<Iter, const T&>
//   constexpr OutputIterator      // constexpr after C++17
//   fill_n(Iter first, Size n, const T& value);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "user_defined_integral.h"

#if TEST_STD_VER > 17
TEST_CONSTEXPR bool test_constexpr() {
    const size_t N = 5;
    int ib[] = {0, 0, 0, 0, 0, 0}; // one bigger than N

    auto it = std::fill_n(std::begin(ib), N, 5);
    return it == (std::begin(ib) + N)
        && std::all_of(std::begin(ib), it, [](int a) {return a == 5; })
        && *it == 0 // don't overwrite the last value in the output array
        ;
    }
#endif

typedef UserDefinedIntegral<unsigned> UDI;

template <class Iter>
void
test_char()
{
    char a[4] = {};
    Iter it = std::fill_n(Iter(a), UDI(4), char(1));
    assert(base(it) == a + 4);
    assert(a[0] == 1);
    assert(a[1] == 1);
    assert(a[2] == 1);
    assert(a[3] == 1);
}

template <class Iter>
void
test_int()
{
    int a[4] = {};
    Iter it = std::fill_n(Iter(a), UDI(4), 1);
    assert(base(it) == a + 4);
    assert(a[0] == 1);
    assert(a[1] == 1);
    assert(a[2] == 1);
    assert(a[3] == 1);
}

void
test_int_array()
{
    int a[4] = {};
    assert(std::fill_n(a, UDI(4), static_cast<char>(1)) == a + 4);
    assert(a[0] == 1);
    assert(a[1] == 1);
    assert(a[2] == 1);
    assert(a[3] == 1);
}

struct source {
    source() : i(0) { }

    operator int() const { return i++; }
    mutable int i;
};

void
test_int_array_struct_source()
{
    int a[4] = {};
    assert(std::fill_n(a, UDI(4), source()) == a + 4);
    assert(a[0] == 0);
    assert(a[1] == 1);
    assert(a[2] == 2);
    assert(a[3] == 3);
}

struct test1 {
    test1() : c(0) { }
    test1(char xc) : c(xc + 1) { }
    char c;
};

void
test_struct_array()
{
    test1 test1a[4] = {};
    assert(std::fill_n(test1a, UDI(4), static_cast<char>(10)) == test1a + 4);
    assert(test1a[0].c == 11);
    assert(test1a[1].c == 11);
    assert(test1a[2].c == 11);
    assert(test1a[3].c == 11);
}

class A
{
    char a_;
public:
    A() {}
    explicit A(char a) : a_(a) {}
    operator unsigned char() const {return 'b';}

    friend bool operator==(const A& x, const A& y)
        {return x.a_ == y.a_;}
};

void
test5()
{
    A a[3];
    assert(std::fill_n(&a[0], UDI(3), A('a')) == a+3);
    assert(a[0] == A('a'));
    assert(a[1] == A('a'));
    assert(a[2] == A('a'));
}

struct Storage
{
  union
  {
    unsigned char a;
    unsigned char b;
  };
};

void test6()
{
  Storage foo[5];
  std::fill_n(&foo[0], UDI(5), Storage());
}


int main(int, char**)
{
    test_char<cpp17_output_iterator<char*> >();
    test_char<forward_iterator<char*> >();
    test_char<bidirectional_iterator<char*> >();
    test_char<random_access_iterator<char*> >();
    test_char<char*>();

    test_int<cpp17_output_iterator<int*> >();
    test_int<forward_iterator<int*> >();
    test_int<bidirectional_iterator<int*> >();
    test_int<random_access_iterator<int*> >();
    test_int<int*>();

    test_int_array();
    test_int_array_struct_source();
    test_struct_array();

    test5();
    test6();

#if TEST_STD_VER > 17
    static_assert(test_constexpr());
#endif

  return 0;
}
