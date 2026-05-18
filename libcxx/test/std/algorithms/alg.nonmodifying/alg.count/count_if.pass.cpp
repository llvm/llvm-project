//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, Predicate<auto, Iter::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr Iter::difference_type   // constexpr after C++17
//   count_if(Iter first, Iter last, Pred pred);

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <functional>
#include <iterator>
#include <list>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"

struct eq {
  TEST_CONSTEXPR eq(int val) : v(val) {}
  TEST_CONSTEXPR bool operator()(int v2) const { return v == v2; }
  int v;
};

#if TEST_STD_VER > 17
TEST_CONSTEXPR bool test_constexpr() {
  int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
  int ib[] = {1, 2, 3, 4, 5, 6};
  return (std::count_if(std::begin(ia), std::end(ia), eq(2)) == 3) &&
         (std::count_if(std::begin(ib), std::end(ib), eq(9)) == 0);
}
#endif

void test_segmented_iterators() {
  {
    // Verify that segmented deque iterators work properly
    const int sizes[] = {0, 1, 2, 1023, 1024, 1025, 2047, 2048, 2049};
    for (const int size : sizes) {
      std::deque<int> deque(size, 1);

      std::ptrdiff_t twos = 0;
      for (int i = 0; i < size; i += 3) {
        deque[i] = 2;
        ++twos;
      }
      std::ptrdiff_t ones = deque.size() - twos;

      assert(std::count_if(deque.begin(), deque.end(), eq(1)) == ones);
      assert(std::count_if(deque.begin(), deque.end(), eq(2)) == twos);
      assert(std::count_if(deque.begin(), deque.end(), eq(99)) == 0);
    }
  }

#if TEST_STD_VER >= 20
  {
    // Verify that join_view of lists work properly
    std::list<std::list<int>> list = {{}, {0}, {1, 2}, {}, {0, 1, 2}, {0, 1, 2, 0}, {1}, {2, 0, 1}};
    auto joined                    = list | std::views::join;

    assert(std::count_if(joined.begin(), joined.end(), eq(0)) == 5);
    assert(std::count_if(joined.begin(), joined.end(), eq(1)) == 5);
    assert(std::count_if(joined.begin(), joined.end(), eq(2)) == 4);
    assert(std::count_if(joined.begin(), joined.end(), eq(99)) == 0);
  }

  {
    // Verify that join_view of vectors work properly
    std::vector<std::vector<int>> vector = {{}, {0}, {1, 2}, {}, {0, 1, 2}, {0, 1, 2, 0}, {1}, {2, 0, 1}};
    auto joined                          = vector | std::views::join;

    assert(std::count_if(joined.begin(), joined.end(), eq(0)) == 5);
    assert(std::count_if(joined.begin(), joined.end(), eq(1)) == 5);
    assert(std::count_if(joined.begin(), joined.end(), eq(2)) == 4);
    assert(std::count_if(joined.begin(), joined.end(), eq(99)) == 0);
  }
#endif // TEST_STD_VER >= 20
}

int main(int, char**) {
  int ia[]          = {0, 1, 2, 2, 0, 1, 2, 3};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  assert(std::count_if(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia + sa), eq(2)) == 3);
  assert(std::count_if(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia + sa), eq(7)) == 0);
  assert(std::count_if(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia), eq(2)) == 0);

#if TEST_STD_VER > 17
  static_assert(test_constexpr());
#endif
#if TEST_STD_VER >= 11
  test_segmented_iterators();
#endif

  return 0;
}
