//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <memory>

// template <class ForwardIt>
// void uninitialized_default_construct(ForwardIt, ForwardIt);

#include <memory>
#include <cstdlib>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

struct Counted {
  static int count;
  static int constructed;
  static void reset() { count = constructed =  0; }
  explicit Counted() { ++count; ++constructed; }
  Counted(Counted const&) { assert(false); }
  ~Counted() { assert(count > 0); --count; }
  friend void operator&(Counted) = delete;
};
int Counted::count = 0;
int Counted::constructed = 0;


struct ThrowsCounted {
  static int count;
  static int constructed;
  static int throw_after;
  static void reset() { throw_after = count = constructed =  0; }
  explicit ThrowsCounted() {
      ++constructed;
      if (throw_after > 0 && --throw_after == 0) {
          TEST_THROW(1);
      }
      ++count;
  }
  ThrowsCounted(ThrowsCounted const&) { assert(false); }
  ~ThrowsCounted() { assert(count > 0); --count; }
  friend void operator&(ThrowsCounted) = delete;
};
int ThrowsCounted::count = 0;
int ThrowsCounted::constructed = 0;
int ThrowsCounted::throw_after = 0;

void test_ctor_throws()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    using It = forward_iterator<ThrowsCounted*>;
    const int N = 5;
    alignas(ThrowsCounted) char pool[sizeof(ThrowsCounted)*N] = {};
    ThrowsCounted* p = (ThrowsCounted*)pool;
    try {
        ThrowsCounted::throw_after = 4;
        std::uninitialized_default_construct_n(It(p), N);
        assert(false);
    } catch (...) {}
    assert(ThrowsCounted::count == 0);
    assert(ThrowsCounted::constructed == 4); // Fourth construction throws
#endif
}

void test_counted()
{
    using It = forward_iterator<Counted*>;
    const int N = 5;
    alignas(Counted) char pool[sizeof(Counted)*N] = {};
    Counted* p = (Counted*)pool;
    It e = std::uninitialized_default_construct_n(It(p), 1);
    assert(e == It(p+1));
    assert(Counted::count == 1);
    assert(Counted::constructed == 1);
    e = std::uninitialized_default_construct_n(It(p+1), 4);
    assert(e == It(p+N));
    assert(Counted::count == 5);
    assert(Counted::constructed == 5);
    std::destroy(p, p+N);
    assert(Counted::count == 0);
}

TEST_CONSTEXPR_CXX26 bool test() {
  struct IntWrapper {
    int n_ = 42;
  };

  {
    constexpr int n = 3;
    std::allocator<int> alloc;
    auto data = alloc.allocate(n);

    auto result = std::uninitialized_default_construct_n(data, n);
    assert(result == data + n);
    for (int i = 0; i != n; ++i) {
      data[i] = -i;
      assert(data[i] == -i);
    }

    std::destroy_n(data, n);
    alloc.deallocate(data, n);
  }
  {
    constexpr int n = 3;
    std::allocator<IntWrapper> alloc;
    auto data = alloc.allocate(n);

    auto result = std::uninitialized_default_construct_n(data, n);
    assert(result == data + n);
    for (int i = 0; i != n; ++i)
      assert(data[i].n_ == 42);

    std::destroy_n(data, n);
    alloc.deallocate(data, n);
  }

  {
    using It        = forward_iterator<int*>;
    constexpr int n = 5;
    int pool[n]{-1, -1, -1, -1, -1};
    int* p    = pool;
    auto end1 = std::uninitialized_default_construct_n(It(p), 1);
    assert(end1 == It(p + 1));
    auto end2 = std::uninitialized_default_construct_n(It(p + 1), 4);
    assert(end2 == It(p + n));

    for (int i = 0; i != n; ++i) {
      pool[i] = i + 17;
      assert(pool[i] == i + 17);
    }
  }
  {
    using It        = forward_iterator<IntWrapper*>;
    constexpr int n = 5;
    IntWrapper pool[n]{IntWrapper{-31}, IntWrapper{-41}, IntWrapper{-59}, IntWrapper{-26}, IntWrapper{-53}};
    IntWrapper* p = pool;
    auto end1     = std::uninitialized_default_construct_n(It(p), 1);
    assert(end1 == It(p + 1));
    assert(pool[0].n_ == 42);
    assert(pool[1].n_ == -41);
    auto end2 = std::uninitialized_default_construct_n(It(p + 1), 4);
    assert(end2 == It(p + n));
    assert(pool[1].n_ == 42);
    assert(pool[2].n_ == 42);
    assert(pool[3].n_ == 42);
    assert(pool[4].n_ == 42);
  }

  return true;
}

#if TEST_STD_VER >= 26
// Test that std::uninitialized_default_construct_n initializes int elements to indeterminate values,
// and thus can cause constant evaluation failure.

enum class assign_elements : bool { no, yes };

constexpr int test_indeterminate_values_helper(assign_elements choice) {
  constexpr int n = 3;

  std::allocator<int> alloc;
  auto data = alloc.allocate(n);
  std::uninitialized_default_construct_n(data, n);

  if (choice == assign_elements::yes)
    for (int i = 0; i != n; ++i)
      data[i] = i + 1;

  int sum = 0;
  for (int i = 0; i != n; ++i)
    sum += data[i];

  std::destroy_n(data, n);
  alloc.deallocate(data, n);

  return sum;
}

template <assign_elements Choice>
concept test_indeterminate_values_result =
    requires { typename std::integral_constant<int, test_indeterminate_values_helper(Choice)>; };

static_assert(!test_indeterminate_values_result<assign_elements::no>);
static_assert(test_indeterminate_values_result<assign_elements::yes>);
#endif // TEST_STD_VER >= 26

int main(int, char**)
{
    test_counted();
    test_ctor_throws();

    test();
#if TEST_STD_VER >= 26
    static_assert(test());
#endif // TEST_STD_VER >= 26

    return 0;
}
