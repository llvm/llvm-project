//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class Iter, IntegralLike Size, class T>
//   requires OutputIterator<Iter, const T&>
//   OutputIterator
//   fill_n(Iter first, Size n, const T& value);

#include <algorithm>
#include <cassert>

#include "test_iterators.h"
#include "user_defined_integral.hpp"

typedef UserDefinedIntegral<unsigned> UDI;

template <class Iter>
void
test_char()
{
    const unsigned n = 4;
    char ca[n] = {0};
    assert(std::fill_n(Iter(ca), UDI(n), char(1)) == std::next(Iter(ca), n));
    assert(ca[0] == 1);
    assert(ca[1] == 1);
    assert(ca[2] == 1);
    assert(ca[3] == 1);
}

template <class Iter>
void
test_int()
{
    const unsigned n = 4;
    int ia[n] = {0};
    assert(std::fill_n(Iter(ia), UDI(n), 1) == std::next(Iter(ia), n));
    assert(ia[0] == 1);
    assert(ia[1] == 1);
    assert(ia[2] == 1);
    assert(ia[3] == 1);
}

void
test_int_array()
{
    const unsigned n = 4;
    int ia[n] = {0};
    assert(std::fill_n(ia, UDI(n), static_cast<char>(1)) == std::next(ia, n));
    assert(ia[0] == 1);
    assert(ia[1] == 1);
    assert(ia[2] == 1);
    assert(ia[3] == 1);
}

struct source {
    source() : i(0) { }

    operator int() const { return i++; }
    mutable int i;
};

void
test_int_array_struct_source()
{
    const unsigned n = 4;
    int ia[n] = {0};
    assert(std::fill_n(ia, UDI(n), source()) == std::next(ia, n));
    assert(ia[0] == 0);
    assert(ia[1] == 1);
    assert(ia[2] == 2);
    assert(ia[3] == 3);
}

struct test1 {
    test1() : c(0) { }
    test1(char c) : c(c + 1) { }
    char c;
};

void
test_struct_array()
{
    const unsigned n = 4;
    test1 test1a[n] = {0};
    assert(std::fill_n(test1a, UDI(n), static_cast<char>(10)) == std::next(test1a, n));
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


int main()
{
    test_char<forward_iterator<char*> >();
    test_char<bidirectional_iterator<char*> >();
    test_char<random_access_iterator<char*> >();
    test_char<char*>();

    test_int<forward_iterator<int*> >();
    test_int<bidirectional_iterator<int*> >();
    test_int<random_access_iterator<int*> >();
    test_int<int*>();

    test_int_array();
    test_int_array_struct_source();
    test_struct_array();

    test5();
    test6();
}
