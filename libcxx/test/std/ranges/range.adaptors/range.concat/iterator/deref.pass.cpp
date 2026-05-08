//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// constexpr decltype(auto) operator*() const

// REQUIRES: std-at-least-c++26

#include <array>
#include <cassert>
#include <ranges>
#include <string>
#include <vector>

#include "test_iterators.h"
#include "../types.h"

template <class Iter, class ValueType = int, class Sent = sentinel_wrapper<Iter>>
constexpr void test() {
  {
    // test with one view
    using View           = minimal_view<Iter, Sent>;
    using ConcatView     = std::ranges::concat_view<View>;
    using ConcatIterator = std::ranges::iterator_t<ConcatView>;

    auto make_concat_view = [](auto begin, auto end) {
      View view{Iter(begin), Sent(Iter(end))};
      return ConcatView(std::move(view));
    };

    std::array array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ConcatView view     = make_concat_view(array.data(), array.data() + array.size());
    ConcatIterator iter = view.begin();
    int& result         = *iter;
    ASSERT_SAME_TYPE(int&, decltype(*iter));
    assert(&result == array.data());
  }

  {
    // test with more than one view
    std::array<int, 3> array1{0, 1, 2};
    std::array<int, 3> array2{0, 1, 2};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    decltype(auto) it1 = view.begin();
    decltype(auto) it2 = view.begin() + 3;

    ASSERT_SAME_TYPE(int&, decltype(*it1));
    assert(*it1 == *it2);
  }

  {
    // constness
    constexpr static std::array<int, 3> array1{0, 1, 2};
    constexpr static std::array<int, 3> array2{0, 1, 2};
    constexpr static std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    decltype(auto) it1 = view.begin();
    decltype(auto) it2 = view.begin() + 3;

    ASSERT_SAME_TYPE(const int&, decltype(*it1));
    assert(*it1 == *it2);
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  {
    // test with more than one view of different types
    std::vector<int> array1{0, 1, 2};
    std::array<int, 3> array2{3, 4, 5};
    std::ranges::concat_view cv(std::views::all(array1), std::views::all(array2));
    decltype(auto) it1 = cv.begin();
    decltype(auto) it2 = cv.begin();

    ASSERT_SAME_TYPE(int&, decltype(*it1));
    it2++;
    assert(*it1 == 0);
    assert(*it2 == 1);
    it2++;
    it2++;
    assert(*it2 == 3);
  }

  {
    // test concat-reference-t
    {
      //  const std::string&  +  std::string_view  +  const std::string&
      const std::array<std::string, 2> left  = {"L0", "L1"};
      std::array<std::string_view, 1> mid    = {std::string_view{"M0"}};
      const std::array<std::string, 1> right = {"R0"};

      auto v   = std::views::concat(left, mid, right);
      auto it  = v.begin();
      auto cit = std::as_const(v).begin();

      //  Common reference of {const std::string&, std::string_view&, const std::string&}  ==>  std::string_view
      static_assert(std::is_same_v<decltype(*it), std::string_view>);
      static_assert(std::is_same_v<decltype(*cit), std::string_view>);

      assert(*it == "L0");
      assert(*std::next(it, 1) == "L1");
      assert(*std::next(it, 2) == "M0");
      assert(*std::next(it, 3) == "R0");
    }

    {
      // std::string&  +  std::string (prvalue)  +  const std::string&
      std::array<std::string, 1> left        = {"L"};
      std::vector<std::string> mid           = {"M"};
      const std::array<std::string, 1> right = {"R"};

      auto mid_prvalue = mid | std::views::transform([](const std::string& s) { return s; });

      auto v   = std::views::concat(left, mid_prvalue, right);
      auto it  = v.begin();
      auto cit = std::as_const(v).begin();

      // Common reference of {std::string&, std::string (prvalue), const std::string&}  ==>  const std::string
      static_assert(std::is_same_v<decltype(*it), const std::string>);
      static_assert(std::is_same_v<decltype(*cit), const std::string>);

      assert(*it == "L");
      assert(*std::next(it, 1) == "M");
      assert(*std::next(it, 2) == "R");
    }
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
