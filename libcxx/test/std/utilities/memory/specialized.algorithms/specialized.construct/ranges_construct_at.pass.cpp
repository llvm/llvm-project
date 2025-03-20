//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>
//
// namespace ranges {
//   template<class T, class... Args>
//     constexpr T* construct_at(T* location, Args&&... args); // since C++20
// }

#include <cassert>
#include <concepts>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

// TODO(varconst): consolidate the ADL checks into a single file.
// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::construct_at)>);

struct Foo {
  int x = 0;
  int y = 0;

  constexpr Foo() = default;
  constexpr explicit Foo(int set_x, int set_y) : x(set_x), y(set_y) {}
  constexpr Foo(std::initializer_list<int>);

  void operator&() const = delete;
  void operator,(auto&&) const = delete;
};

ASSERT_SAME_TYPE(decltype(std::ranges::construct_at((int*)nullptr)), int*);
ASSERT_SAME_TYPE(decltype(std::ranges::construct_at((Foo*)nullptr)), Foo*);

struct Counted {
  int& count;

  constexpr Counted(int& count_ref) : count(count_ref) { ++count; }
  constexpr Counted(const Counted& rhs) : count(rhs.count) { ++count; }
  constexpr ~Counted() { --count; }
};

struct CountDefaultInitializations {
  CountDefaultInitializations() { ++constructions; }
  static int constructions;
};
int CountDefaultInitializations::constructions = 0;

constexpr bool test() {
  // Value initialization.
  {
    int x = 1;

    std::same_as<int*> auto result = std::ranges::construct_at(&x);
    assert(result == &x);
    assert(x == 0);
  }

  // Copy initialization.
  {
    int x = 1;

    std::same_as<int*> auto result = std::ranges::construct_at(&x, 42);
    assert(result == &x);
    assert(x == 42);
  }

  // Explicit multiargument constructor; also checks that the initializer list constructor is not invoked.
  {
    Foo f;

    std::same_as<Foo*> auto result = std::ranges::construct_at(std::addressof(f), 42, 123);
    assert(result == std::addressof(f));
    assert(f.x == 42);
    assert(f.y == 123);
  }

  // Works with buffers of uninitialized memory.
  {
    std::allocator<Counted> alloc;
    Counted* out = alloc.allocate(2);
    int count    = 0;

    std::same_as<Counted*> auto result = std::ranges::construct_at(out, count);
    assert(result == out);
    assert(count == 1);

    result = std::ranges::construct_at(out + 1, count);
    assert(result == out + 1);
    assert(count == 2);

    std::destroy(out, out + 1);
    alloc.deallocate(out, 2);
  }

  // Test LWG3436, std::ranges::construct_at with array types
  {
    {
      using Array = int[1];
      Array array;
      std::same_as<Array*> auto result = std::ranges::construct_at(&array);
      assert(result == &array);
      assert(array[0] == 0);
    }
    {
      using Array = int[2];
      Array array;
      std::same_as<Array*> auto result = std::ranges::construct_at(&array);
      assert(result == &array);
      assert(array[0] == 0);
      assert(array[1] == 0);
    }
    {
      using Array = int[3];
      Array array;
      std::same_as<Array*> auto result = std::ranges::construct_at(&array);
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
        std::ranges::construct_at(&array);
        assert(CountDefaultInitializations::constructions == 1);
      }
      {
        using Array = CountDefaultInitializations[2];
        CountDefaultInitializations array[2];
        CountDefaultInitializations::constructions = 0;
        std::ranges::construct_at(&array);
        assert(CountDefaultInitializations::constructions == 2);
      }
      {
        using Array = CountDefaultInitializations[3];
        CountDefaultInitializations array[3];
        CountDefaultInitializations::constructions = 0;
        std::ranges::construct_at(&array);
        assert(CountDefaultInitializations::constructions == 3);
      }
    }
  }

  return true;
}

template <class... Args>
constexpr bool can_construct_at = requires { std::ranges::construct_at(std::declval<Args>()...); };

struct NoDefault {
  NoDefault() = delete;
};

// Check that SFINAE works.
static_assert(can_construct_at<Foo*, int, int>);
static_assert(!can_construct_at<Foo*, int>);
static_assert(!can_construct_at<Foo*, int, int, int>);
static_assert(!can_construct_at<std::nullptr_t, int, int>);
static_assert(!can_construct_at<int*, int, int>);
static_assert(!can_construct_at<contiguous_iterator<Foo*>, int, int>);
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
