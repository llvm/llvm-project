//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS(gcc): -Wno-bool-compare
// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-sign-compare
// MSVC warning C4389: '==': signed/unsigned mismatch
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4389

// <algorithm>

// template<InputIterator Iter, class T>
//   requires HasEqualTo<Iter::value_type, T>
//   constexpr Iter   // constexpr after C++17
//   find(Iter first, Iter last, const T& value);

#include <algorithm>
#include <cassert>
#include <deque>
#include <vector>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

static std::vector<int> comparable_data;

template <class ArrayT, class CompareT>
struct Test {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    ArrayT arr[] = {
        ArrayT(1), ArrayT(2), ArrayT(3), ArrayT(4), ArrayT(5), ArrayT(6), ArrayT(7), ArrayT(8), ArrayT(9), ArrayT(10)};

    static_assert(std::is_same<decltype(std::find(Iter(arr), Iter(arr), 0)), Iter>::value, "");

    { // first element matches
      Iter iter = std::find(Iter(arr), Iter(arr + 10), CompareT(1));
      assert(*iter == ArrayT(1));
      assert(base(iter) == arr);
    }

    { // range is empty; return last
      Iter iter = std::find(Iter(arr), Iter(arr), CompareT(1));
      assert(base(iter) == arr);
    }

    { // if multiple elements match, return the first match
      ArrayT data[] = {
          ArrayT(1), ArrayT(2), ArrayT(3), ArrayT(4), ArrayT(5), ArrayT(6), ArrayT(7), ArrayT(5), ArrayT(4)};
      Iter iter = std::find(Iter(std::begin(data)), Iter(std::end(data)), CompareT(5));
      assert(*iter == ArrayT(5));
      assert(base(iter) == data + 4);
    }

    { // some element matches
      Iter iter = std::find(Iter(arr), Iter(arr + 10), CompareT(6));
      assert(*iter == ArrayT(6));
      assert(base(iter) == arr + 5);
    }

    { // last element matches
      Iter iter = std::find(Iter(arr), Iter(arr + 10), CompareT(10));
      assert(*iter == ArrayT(10));
      assert(base(iter) == arr + 9);
    }

    { // if no element matches, last is returned
      Iter iter = std::find(Iter(arr), Iter(arr + 10), CompareT(20));
      assert(base(iter) == arr + 10);
    }

    if (!TEST_IS_CONSTANT_EVALUATED)
      comparable_data.clear();
  }
};

template <class IndexT>
class Comparable {
  IndexT index_;

  static IndexT init_index(IndexT i) {
    IndexT size = static_cast<IndexT>(comparable_data.size());
    comparable_data.push_back(i);
    return size;
  }

public:
  Comparable(IndexT i) : index_(init_index(i)) {}

  friend bool operator==(const Comparable& lhs, const Comparable& rhs) {
    return comparable_data[lhs.index_] == comparable_data[rhs.index_];
  }
};

#if TEST_STD_VER >= 20
template <class ElementT>
class TriviallyComparable {
  ElementT el_;

public:
  explicit constexpr TriviallyComparable(ElementT el) : el_(el) {}
  bool operator==(const TriviallyComparable&) const = default;
};
#endif

template <class CompareT>
struct TestTypes {
  template <class T>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(types::cpp17_input_iterator_list<T*>(), Test<T, CompareT>());
  }
};

void test_deque() {
  { // empty deque
    std::deque<int> data;
    assert(std::find(data.begin(), data.end(), 4) == data.end());
  }

  { // single element - match
    std::deque<int> data;
    data.push_back(4);
    assert(std::find(data.begin(), data.end(), 4) == data.begin());
  }

  { // single element - no match
    std::deque<int> data;
    data.push_back(3);
    assert(std::find(data.begin(), data.end(), 4) == data.end());
  }

  // many elements
  int sizes[] = {2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
  for (auto size : sizes) {
    { // last element match
      std::deque<int> data;
      data.resize(size);
      std::fill(data.begin(), data.end(), 3);
      data[size - 1] = 4;
      assert(std::find(data.begin(), data.end(), 4) == data.end() - 1);
    }

    { // second-last element match
      std::deque<int> data;
      data.resize(size);
      std::fill(data.begin(), data.end(), 3);
      data[size - 2] = 4;
      assert(std::find(data.begin(), data.end(), 4) == data.end() - 2);
    }

    { // no match
      std::deque<int> data;
      data.resize(size);
      std::fill(data.begin(), data.end(), 3);
      assert(std::find(data.begin(), data.end(), 4) == data.end());
    }
  }
}

template <class T>
struct TestIntegerPromotions1 {
  template <class U>
  TEST_CONSTEXPR_CXX20 void test(T val, U to_find) {
    bool expect_match = val == to_find;
    assert(std::find(&val, &val + 1, to_find) == (expect_match ? &val : &val + 1));
  }

  template <class U>
  TEST_CONSTEXPR_CXX20 void operator()() {
    test<U>(0, 0);
    test<U>(0, 1);
    test<U>(1, 1);
    test<U>(0, -1);
    test<U>(-1, -1);
    test<U>(0, U(-127));
    test<U>(T(-127), U(-127));
    test<U>(T(-128), U(-128));
    test<U>(T(-129), U(-129));
    test<U>(T(255), U(255));
    test<U>(T(256), U(256));
    test<U>(T(257), U(257));
    test<U>(0, std::numeric_limits<U>::min());
    test<U>(T(std::numeric_limits<U>::min()), std::numeric_limits<U>::min());
    test<U>(0, std::numeric_limits<U>::min() + 1);
    test<U>(T(std::numeric_limits<U>::min() + 1), std::numeric_limits<U>::min() + 1);
    test<U>(0, std::numeric_limits<U>::max());
    test<U>(T(std::numeric_limits<U>::max()), std::numeric_limits<U>::max());
    test<U>(T(std::numeric_limits<U>::max() - 1), std::numeric_limits<U>::max() - 1);
  }
};

struct TestIntegerPromotions {
  template <class T>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(types::integral_types(), TestIntegerPromotions1<T>());
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::integer_types(), TestTypes<char>());
  types::for_each(types::integer_types(), TestTypes<int>());
  types::for_each(types::integer_types(), TestTypes<long long>());
#if TEST_STD_VER >= 20
  Test<TriviallyComparable<char>, TriviallyComparable<char>>().operator()<TriviallyComparable<char>*>();
  Test<TriviallyComparable<wchar_t>, TriviallyComparable<wchar_t>>().operator()<TriviallyComparable<wchar_t>*>();
#endif

  // TODO: Remove the `_LIBCPP_ENABLE_EXPERIMENTAL` check once we have the FTM guarded or views::join isn't
  // experimental anymore
#if TEST_STD_VER >= 20 && (!defined(_LIBCPP_VERSION) || defined(_LIBCPP_ENABLE_EXPERIMENTAL))
  {
    std::vector<std::vector<int>> vec = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto view                         = vec | std::views::join;
    assert(std::find(view.begin(), view.end(), 4) == std::next(view.begin(), 3));
  }
#endif

  types::for_each(types::integral_types(), TestIntegerPromotions());

  return true;
}

int main(int, char**) {
  test_deque();
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  Test<Comparable<char>, Comparable<char> >().operator()<Comparable<char>*>();
  Test<Comparable<wchar_t>, Comparable<wchar_t> >().operator()<Comparable<wchar_t>*>();

  return 0;
}
