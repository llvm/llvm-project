//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// template <nothrow-forward-iterator ForwardIterator, nothrow-sentinel-for<ForwardIterator> Sentinel>
//   requires default_initializable<iter_value_t<ForwardIterator>>
// ForwardIterator ranges::uninitialized_default_construct(ForwardIterator first, Sentinel last);
//
// template <nothrow-forward-range ForwardRange>
//   requires default_initializable<range_value_t<ForwardRange>>
// borrowed_iterator_t<ForwardRange> ranges::uninitialized_default_construct(ForwardRange&& range);

#include <cassert>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>

#include "../buffer.h"
#include "../counted.h"
#include "test_macros.h"
#include "test_iterators.h"

// TODO(varconst): consolidate the ADL checks into a single file.
// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::uninitialized_default_construct)>);

struct NotDefaultCtrable { NotDefaultCtrable() = delete; };
static_assert(!std::is_invocable_v<decltype(std::ranges::uninitialized_default_construct),
    NotDefaultCtrable*, NotDefaultCtrable*>);

TEST_CONSTEXPR_CXX26 bool test() {
  struct IntWrapper {
    int n_ = 42;
  };

  // (iter, sentinel) overload.
  {
    constexpr int n = 3;
    std::allocator<int> alloc;
    auto data = alloc.allocate(n);
    auto last = data + n;

    std::ranges::uninitialized_default_construct(data, last);
    for (int i = 0; i != n; ++i) {
      data[i] = -i;
      assert(data[i] == -i);
    }

    std::ranges::destroy(data, last);
    alloc.deallocate(data, n);
  }
  // (range) overload.
  {
    constexpr int n = 3;
    std::allocator<int> alloc;
    auto data = alloc.allocate(n);
    auto r    = std::span<int>{data, n};

    std::ranges::uninitialized_default_construct(r);
    for (int i = 0; i != n; ++i) {
      r[i] = -i;
      assert(r[i] == -i);
    }

    std::ranges::destroy(r);
    alloc.deallocate(data, n);
  }
  // (iter, sentinel) overload.
  {
    constexpr int n = 3;
    std::allocator<IntWrapper> alloc;
    auto data = alloc.allocate(n);
    auto last = data + n;

    std::ranges::uninitialized_default_construct(data, last);
    for (int i = 0; i != n; ++i)
      assert(data[i].n_ == 42);

    std::ranges::destroy(data, last);
    alloc.deallocate(data, n);
  }
  // (range) overload.
  {
    constexpr int n = 3;
    std::allocator<IntWrapper> alloc;
    auto data = alloc.allocate(n);
    auto r    = std::span<IntWrapper>{data, n};

    std::ranges::uninitialized_default_construct(r);
    for (int i = 0; i != n; ++i)
      assert(r[i].n_ == 42);

    std::ranges::destroy(r);
    alloc.deallocate(data, n);
  }

  // (iter, sentinel) overload.
  {
    using It        = forward_iterator<int*>;
    constexpr int n = 5;
    int pool[n]{-1, -1, -1, -1, -1};
    int* p    = pool;
    int* pend = p + n;

    std::ranges::uninitialized_default_construct(It(p), It(pend));

    for (int i = 0; i != n; ++i) {
      pool[i] = i + 17;
      assert(pool[i] == i + 17);
    }
  }
  // (range) overload.
  {
    using It        = forward_iterator<int*>;
    constexpr int n = 5;
    int pool[n]{-1, -1, -1, -1, -1};
    int* p    = pool;
    int* pend = p + n;

    std::ranges::uninitialized_default_construct(std::ranges::subrange(It(p), It(pend)));

    for (int i = 0; i != n; ++i) {
      pool[i] = i + 17;
      assert(pool[i] == i + 17);
    }
  }
  // (iter, sentinel) overload.
  {
    using It        = forward_iterator<IntWrapper*>;
    constexpr int n = 5;
    IntWrapper pool[n]{IntWrapper{-31}, IntWrapper{-41}, IntWrapper{-59}, IntWrapper{-26}, IntWrapper{-53}};
    IntWrapper* p = pool;
    std::ranges::uninitialized_default_construct(It(p), It(p + 1));
    assert(pool[0].n_ == 42);
    assert(pool[1].n_ == -41);
    std::ranges::uninitialized_default_construct(It(p + 1), It(p + n));
    assert(pool[1].n_ == 42);
    assert(pool[2].n_ == 42);
    assert(pool[3].n_ == 42);
    assert(pool[4].n_ == 42);
  }
  // (range) overload.
  {
    using It        = forward_iterator<IntWrapper*>;
    constexpr int n = 5;
    IntWrapper pool[n]{IntWrapper{-31}, IntWrapper{-41}, IntWrapper{-59}, IntWrapper{-26}, IntWrapper{-53}};
    IntWrapper* p = pool;
    std::ranges::uninitialized_default_construct(std::ranges::subrange(It(p), It(p + 1)));
    assert(pool[0].n_ == 42);
    assert(pool[1].n_ == -41);
    std::ranges::uninitialized_default_construct(std::ranges::subrange(It(p + 1), It(p + n)));
    assert(pool[1].n_ == 42);
    assert(pool[2].n_ == 42);
    assert(pool[3].n_ == 42);
    assert(pool[4].n_ == 42);
  }

  return true;
}

#if TEST_STD_VER >= 26
// Test that std::ranges::uninitialized_default_construct initializes int elements to indeterminate values,
// and thus can cause constant evaluation failure.

enum class assign_elements : bool { no, yes };
enum class algorithm_style { iterator_sentinel, range };

constexpr int test_indeterminate_values_helper(algorithm_style style, assign_elements choice) {
  constexpr int n = 3;

  std::allocator<int> alloc;
  auto data = alloc.allocate(n);
  auto last = data + n;

  if (style == algorithm_style::range)
    std::ranges::uninitialized_default_construct(std::span<int>{data, n});
  else
    std::ranges::uninitialized_default_construct(data, last);

  if (choice == assign_elements::yes)
    for (int i = 0; i != n; ++i)
      data[i] = i + 1;

  int sum = 0;
  for (int i = 0; i != n; ++i)
    sum += data[i];

  std::ranges::destroy(data, last);
  alloc.deallocate(data, n);

  return sum;
}

template <algorithm_style Style, assign_elements Choice>
concept test_indeterminate_values_result =
    requires { typename std::integral_constant<int, test_indeterminate_values_helper(Style, Choice)>; };

static_assert(!test_indeterminate_values_result<algorithm_style::iterator_sentinel, assign_elements::no>);
static_assert(!test_indeterminate_values_result<algorithm_style::range, assign_elements::no>);
static_assert(test_indeterminate_values_result<algorithm_style::iterator_sentinel, assign_elements::yes>);
static_assert(test_indeterminate_values_result<algorithm_style::range, assign_elements::yes>);
#endif // TEST_STD_VER >= 26

int main(int, char**) {
  // An empty range -- no default constructors should be invoked.
  {
    Buffer<Counted, 1> buf;

    std::ranges::uninitialized_default_construct(buf.begin(), buf.begin());
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);

    std::ranges::uninitialized_default_construct(std::ranges::empty_view<Counted>());
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);

    forward_iterator<Counted*> it(buf.begin());
    auto range = std::ranges::subrange(it, sentinel_wrapper<forward_iterator<Counted*>>(it));
    std::ranges::uninitialized_default_construct(range.begin(), range.end());
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);

    std::ranges::uninitialized_default_construct(range);
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);
  }

  // A range containing several objects, (iter, sentinel) overload.
  {
    constexpr int N = 5;
    Buffer<Counted, 5> buf;

    std::ranges::uninitialized_default_construct(buf.begin(), buf.end());
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // A range containing several objects, (range) overload.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    auto range = std::ranges::subrange(buf.begin(), buf.end());
    std::ranges::uninitialized_default_construct(range);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // Using `counted_iterator`.
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    std::ranges::uninitialized_default_construct(
        std::counted_iterator(buf.begin(), N), std::default_sentinel);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }

  // Using `views::counted`.
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    std::ranges::uninitialized_default_construct(std::views::counted(buf.begin(), N));
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }

  // Using `reverse_view`.
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    auto range = std::ranges::subrange(buf.begin(), buf.begin() + N);
    std::ranges::uninitialized_default_construct(std::ranges::reverse_view(range));
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }

  // An exception is thrown while objects are being created -- the existing objects should stay
  // valid. (iterator, sentinel) overload.
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    Buffer<Counted, 5> buf;

    Counted::throw_on = 3; // When constructing the fourth object (counting from one).
    try {
      std::ranges::uninitialized_default_construct(buf.begin(), buf.end());
    } catch(...) {}
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 3);
    std::destroy(buf.begin(), buf.begin() + Counted::total_objects);
    Counted::reset();
  }

  // An exception is thrown while objects are being created -- the existing objects should stay
  // valid. (range) overload.
  {
    Buffer<Counted, 5> buf;

    Counted::throw_on = 3; // When constructing the fourth object.
    try {
      std::ranges::uninitialized_default_construct(buf);
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
