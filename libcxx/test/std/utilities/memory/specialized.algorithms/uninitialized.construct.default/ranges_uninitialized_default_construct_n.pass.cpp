//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// template <nothrow-forward-iterator ForwardIterator>
//   requires default_initializable<iter_value_t<ForwardIterator>>
// ForwardIterator ranges::uninitialized_default_construct_n(ForwardIterator first,
//     iter_difference_t<ForwardIterator> n);

#include <cassert>
#include <memory>
#include <ranges>
#include <type_traits>

#include "../buffer.h"
#include "../counted.h"
#include "test_macros.h"
#include "test_iterators.h"

// TODO(varconst): consolidate the ADL checks into a single file.
// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::uninitialized_default_construct_n)>);

struct NotDefaultCtrable { NotDefaultCtrable() = delete; };
static_assert(!std::is_invocable_v<decltype(std::ranges::uninitialized_default_construct_n),
    NotDefaultCtrable*, int>);

TEST_CONSTEXPR_CXX26 bool test() {
  struct IntWrapper {
    int n_ = 42;
  };

  {
    constexpr int n = 3;
    std::allocator<int> alloc;
    auto data = alloc.allocate(n);

    auto result = std::ranges::uninitialized_default_construct_n(data, n);
    assert(result == data + n);
    for (int i = 0; i != n; ++i) {
      data[i] = -i;
      assert(data[i] == -i);
    }

    std::ranges::destroy_n(data, n);
    alloc.deallocate(data, n);
  }
  {
    constexpr int n = 3;
    std::allocator<IntWrapper> alloc;
    auto data = alloc.allocate(n);

    auto result = std::ranges::uninitialized_default_construct_n(data, n);
    assert(result == data + n);
    for (int i = 0; i != n; ++i)
      assert(data[i].n_ == 42);

    std::ranges::destroy_n(data, n);
    alloc.deallocate(data, n);
  }

  {
    using It        = forward_iterator<int*>;
    constexpr int n = 5;
    int pool[n]{-1, -1, -1, -1, -1};
    int* p    = pool;
    auto end1 = std::ranges::uninitialized_default_construct_n(It(p), 1);
    assert(end1 == It(p + 1));
    auto end2 = std::ranges::uninitialized_default_construct_n(It(p + 1), 4);
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
    auto end1     = std::ranges::uninitialized_default_construct_n(It(p), 1);
    assert(end1 == It(p + 1));
    assert(pool[0].n_ == 42);
    assert(pool[1].n_ == -41);
    auto end2 = std::ranges::uninitialized_default_construct_n(It(p + 1), 4);
    assert(end2 == It(p + n));
    assert(pool[1].n_ == 42);
    assert(pool[2].n_ == 42);
    assert(pool[3].n_ == 42);
    assert(pool[4].n_ == 42);
  }

  return true;
}

#if TEST_STD_VER >= 26
// Test that std::ranges::uninitialized_default_construct_n initializes int elements to indeterminate values,
// and thus can cause constant evaluation failure.

enum class assign_elements : bool { no, yes };

constexpr int test_indeterminate_values_helper(assign_elements choice) {
  constexpr int n = 3;

  std::allocator<int> alloc;
  auto data = alloc.allocate(n);
  std::ranges::uninitialized_default_construct_n(data, n);

  if (choice == assign_elements::yes)
    for (int i = 0; i != n; ++i)
      data[i] = i + 1;

  int sum = 0;
  for (int i = 0; i != n; ++i)
    sum += data[i];

  std::ranges::destroy_n(data, n);
  alloc.deallocate(data, n);

  return sum;
}

template <assign_elements Choice>
concept test_indeterminate_values_result =
    requires { typename std::integral_constant<int, test_indeterminate_values_helper(Choice)>; };

static_assert(!test_indeterminate_values_result<assign_elements::no>);
static_assert(test_indeterminate_values_result<assign_elements::yes>);
#endif // TEST_STD_VER >= 26

int main(int, char**) {
  // An empty range -- no default constructors should be invoked.
  {
    Buffer<Counted, 1> buf;

    std::ranges::uninitialized_default_construct_n(buf.begin(), 0);
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);
  }

  // A range containing several objects.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    std::ranges::uninitialized_default_construct_n(buf.begin(), N);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // An exception is thrown while objects are being created -- the existing objects should stay
  // valid.
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    Counted::throw_on = 3; // When constructing the fourth object (counting from one).
    try {
      std::ranges::uninitialized_default_construct_n(buf.begin(), N);
    } catch(...) {}
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 3);
    std::destroy(buf.begin(), buf.begin() + Counted::total_objects);
    Counted::reset();
  }
#endif  // TEST_HAS_NO_EXCEPTIONS

  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif // TEST_STD_VER >= 26

  return 0;
}
