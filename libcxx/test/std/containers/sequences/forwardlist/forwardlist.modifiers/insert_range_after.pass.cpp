//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<T> R>
//   constexpr iterator insert_range_after(const_iterator position, R&& rg); // C++23

#include <forward_list>

#include "../../insert_range_sequence_containers.h"
#include "test_macros.h"

template <class Container, class Range>
concept HasInsertRangeAfter = requires (Container& c, Range&& range) {
  c.insert_range_after(c.begin(), range);
};

template <template <class...> class Container, class T, class U>
constexpr bool test_constraints_insert_range_after() {
  // Input range with the same value type.
  static_assert(HasInsertRangeAfter<Container<T>, InputRange<T>>);
  // Input range with a convertible value type.
  static_assert(HasInsertRangeAfter<Container<T>, InputRange<U>>);
  // Input range with a non-convertible value type.
  static_assert(!HasInsertRangeAfter<Container<T>, InputRange<Empty>>);
  // Not an input range.
  static_assert(!HasInsertRangeAfter<Container<T>, InputRangeNotDerivedFrom>);
  static_assert(!HasInsertRangeAfter<Container<T>, InputRangeNotIndirectlyReadable>);
  static_assert(!HasInsertRangeAfter<Container<T>, InputRangeNotInputOrOutputIterator>);

  return true;
}

// Tested cases:
// - different kinds of insertions (inserting an {empty/one-element/mid-sized/long range} into an
//   {empty/one-element/full} container at the {beginning/middle/end});
// - inserting move-only elements;
// - an exception is thrown when copying the elements or when allocating new elements.

template <class T, class Iter, class Sent, class Alloc>
constexpr void test_sequence_insert_range_after() {
  using Container = std::forward_list<T, Alloc>;
  // Index `0` translates to `before_begin()` and the last index translates to the index before `end()`.
  auto get_insert_pos = [](auto& c, auto& test_case) { return std::ranges::next(c.before_begin(), test_case.index); };
  // Unlike `insert_range` in other containers, `insert_range_after` returns the iterator to the last inserted element.
  auto get_return_pos = [](auto& c, auto& test_case) {
    return std::ranges::next(c.before_begin(), test_case.index + test_case.input.size());
  };

  { // Empty container.
    { // empty_c.insert_range_after(end, empty_range)
      auto& test_case = EmptyContainer_EmptyRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // empty_c.insert_range_after(end, one_element_range)
      auto& test_case = EmptyContainer_OneElementRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // empty_c.insert_range_after(end, mid_range)
      auto& test_case = EmptyContainer_MidRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }
  }

  { // One-element container.
    { // one_element_c.insert_range_after(begin, empty_range)
      auto& test_case = OneElementContainer_Begin_EmptyRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // one_element_c.insert_range_after(end, empty_range)
      auto& test_case = OneElementContainer_End_EmptyRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // one_element_c.insert_range_after(begin, one_element_range)
      auto& test_case = OneElementContainer_Begin_OneElementRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // one_element_c.insert_range_after(end, one_element_range)
      auto& test_case = OneElementContainer_End_OneElementRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // one_element_c.insert_range_after(begin, mid_range)
      auto& test_case = OneElementContainer_Begin_MidRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // one_element_c.insert_range_after(end, mid_range)
      auto& test_case = OneElementContainer_End_MidRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }
  }

  { // Full container.
    { // full_container.insert_range_after(begin, empty_range)
      auto& test_case = FullContainer_Begin_EmptyRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(mid, empty_range)
      auto& test_case = FullContainer_Mid_EmptyRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(end, empty_range)
      auto& test_case = FullContainer_End_EmptyRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(begin, one_element_range)
      auto& test_case = FullContainer_Begin_OneElementRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(end, one_element_range)
      auto& test_case = FullContainer_Mid_OneElementRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(end, one_element_range)
      auto& test_case = FullContainer_End_OneElementRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(begin, mid_range)
      auto& test_case = FullContainer_Begin_MidRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(mid, mid_range)
      auto& test_case = FullContainer_Mid_MidRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(end, mid_range)
      auto& test_case = FullContainer_End_MidRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(begin, long_range)
      auto& test_case = FullContainer_Begin_LongRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(mid, long_range)
      auto& test_case = FullContainer_Mid_LongRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }

    { // full_container.insert_range_after(end, long_range)
      auto& test_case = FullContainer_End_LongRange<T>;

      Container c(test_case.initial.begin(), test_case.initial.end());
      auto in = wrap_input<Iter, Sent>(test_case.input);
      auto pos = get_insert_pos(c, test_case);

      auto result = c.insert_range_after(pos, in);
      assert(std::ranges::equal(c, test_case.expected));
      assert(result == get_return_pos(c, test_case));
    }
  }

  // Also check inserting after `begin()` (the tests above only use `before_begin()`).
  {
    Container c{5, 1, 3, 4, 9};
    Buffer<T> input{-18, -15, -11};
    auto in = wrap_input<Iter, Sent>(input);

    auto result = c.insert_range_after(c.begin(), in);
    assert(std::ranges::equal(c, Buffer<int>{5, -18, -15, -11, 1, 3, 4, 9}));
    assert(result == std::ranges::next(c.begin(), 3));
  }
}

void test_sequence_insert_range_after_move_only() {
  MoveOnly input[5];
  std::ranges::subrange in(std::move_iterator{input}, std::move_iterator{input + 5});

  std::forward_list<MoveOnly> c;
  c.insert_range_after(c.before_begin(), in);
}

void test_insert_range_after_exception_safety_throwing_copy() {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  using T = ThrowingCopy<3>;
  T::reset();
  T in[5];

  try {
    std::forward_list<T> c;
    c.insert_range_after(c.before_begin(), in);
    assert(false); // The function call above should throw.

  } catch (int) {
    assert(T::created_by_copying == 3);
    assert(T::destroyed == 2); // No destructor call for the partially-constructed element.
  }
#endif
}

template <class T>
void test_insert_range_after_exception_safety_throwing_allocator() {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  T in[] = {0, 1};

  try {
    ThrowingAllocator<T> alloc;

    globalMemCounter.reset();
    std::forward_list<T, ThrowingAllocator<T>> c(alloc);
    c.insert_range_after(c.before_begin(), in);
    assert(false); // The function call above should throw.

  } catch (int) {
    assert(globalMemCounter.new_called == globalMemCounter.delete_called);
  }
#endif
}

int main(int, char**) {
  static_assert(test_constraints_insert_range_after<std::forward_list, int, double>());

  for_all_iterators_and_allocators<int, const int*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_insert_range_after<int, Iter, Sent, Alloc>();
  });
  test_sequence_insert_range_after_move_only();

  test_insert_range_after_exception_safety_throwing_copy();
  test_insert_range_after_exception_safety_throwing_allocator<int>();

  return 0;
}
