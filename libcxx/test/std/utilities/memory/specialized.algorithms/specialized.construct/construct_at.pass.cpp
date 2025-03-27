//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// template <class T, class ...Args>
// constexpr T* construct_at(T* location, Args&& ...args);

#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>

#include "test_iterators.h"
#include "test_macros.h"

struct Foo {
  constexpr Foo() {}
  constexpr Foo(int a, char b, double c) : a_(a), b_(b), c_(c) {}
  constexpr Foo(int a, char b, double c, int* count) : Foo(a, b, c) { *count += 1; }
  constexpr bool operator==(Foo const& other) const { return a_ == other.a_ && b_ == other.b_ && c_ == other.c_; }

private:
  int a_;
  char b_;
  double c_;
};

struct Counted {
  int& count_;
  constexpr Counted(int& count) : count_(count) { ++count; }
  constexpr Counted(Counted const& that) : count_(that.count_) { ++count_; }
  constexpr ~Counted() { --count_; }
};

struct CountDefaultInitializations {
  CountDefaultInitializations() { ++constructions; }
  static int constructions;
};
int CountDefaultInitializations::constructions = 0;

constexpr bool test() {
  {
    int i                       = 99;
    std::same_as<int*> auto res = std::construct_at(&i);
    assert(res == &i);
    assert(*res == 0);
  }

  {
    int i                       = 0;
    std::same_as<int*> auto res = std::construct_at(&i, 42);
    assert(res == &i);
    assert(*res == 42);
  }

  {
    Foo foo                     = {};
    int count                   = 0;
    std::same_as<Foo*> auto res = std::construct_at(&foo, 42, 'x', 123.89, &count);
    assert(res == &foo);
    assert(*res == Foo(42, 'x', 123.89));
    assert(count == 1);
  }

  {
    std::allocator<Counted> a;
    Counted* p = a.allocate(2);
    int count  = 0;
    std::construct_at(p, count);
    assert(count == 1);
    std::construct_at(p + 1, count);
    assert(count == 2);
    (p + 1)->~Counted();
    assert(count == 1);
    p->~Counted();
    assert(count == 0);
    a.deallocate(p, 2);
  }

  // Test LWG3436, std::construct_at with array types
  {
    {
      using Array = int[1];
      Array array;
      std::same_as<Array*> auto result = std::construct_at(&array);
      assert(result == &array);
      assert(array[0] == 0);
    }
    {
      using Array = int[2];
      Array array;
      std::same_as<Array*> auto result = std::construct_at(&array);
      assert(result == &array);
      assert(array[0] == 0);
      assert(array[1] == 0);
    }
    {
      using Array = int[3];
      Array array;
      std::same_as<Array*> auto result = std::construct_at(&array);
      assert(result == &array);
      assert(array[0] == 0);
      assert(array[1] == 0);
      assert(array[2] == 0);
    }

    // Make sure we initialize the right number of elements. This can't be done inside
    // constexpr since it requires a global variable.
    if (!TEST_IS_CONSTANT_EVALUATED) {
      {
        using Array = CountDefaultInitializations[1];
        CountDefaultInitializations array[1];
        CountDefaultInitializations::constructions = 0;
        std::construct_at(&array);
        assert(CountDefaultInitializations::constructions == 1);
      }
      {
        using Array = CountDefaultInitializations[2];
        CountDefaultInitializations array[2];
        CountDefaultInitializations::constructions = 0;
        std::construct_at(&array);
        assert(CountDefaultInitializations::constructions == 2);
      }
      {
        using Array = CountDefaultInitializations[3];
        CountDefaultInitializations array[3];
        CountDefaultInitializations::constructions = 0;
        std::construct_at(&array);
        assert(CountDefaultInitializations::constructions == 3);
      }
    }
  }

  return true;
}

template <class... Args>
constexpr bool can_construct_at = requires { std::construct_at(std::declval<Args>()...); };

struct NoDefault {
  NoDefault() = delete;
};

// Check that SFINAE works.
static_assert(can_construct_at<int*, int>);
static_assert(can_construct_at<Foo*, int, char, double>);
static_assert(!can_construct_at<Foo*, int, char>);
static_assert(!can_construct_at<Foo*, int, char, double, int>);
static_assert(!can_construct_at<std::nullptr_t, int, char, double>);
static_assert(!can_construct_at<int*, int, char, double>);
static_assert(!can_construct_at<contiguous_iterator<Foo*>, int, char, double>);
// Can't construct function pointers.
static_assert(!can_construct_at<int (*)()>);
static_assert(!can_construct_at<int (*)(), std::nullptr_t>);

// LWG3436
static_assert(can_construct_at<int (*)[3]>);        // test the test
static_assert(!can_construct_at<int (*)[]>);        // unbounded arrays should SFINAE away
static_assert(!can_construct_at<NoDefault (*)[1]>); // non default constructible shouldn't work

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
