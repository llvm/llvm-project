//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// REQUIRES: libcpp-hardening-mode=debug
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// This test uses a specific combination of an invalid comparator and sequence of values to
// ensure that our sorting functions do not go out-of-bounds and satisfy strict weak ordering in that case.
// Instead, we should fail loud with an assertion. The specific issue we're looking for here is when the comparator
// does not satisfy the strict weak ordering:
//
//    Irreflexivity: comp(a, a) is false
//    Antisymmetry: comp(a, b) implies that !comp(b, a)
//    Transitivity: comp(a, b), comp(b, c) imply comp(a, c)
//    Transitivity of equivalence: !comp(a, b), !comp(b, a), !comp(b, c), !comp(c, b) imply !comp(a, c), !comp(c, a)
//
// If this is not satisfied, we have seen issues in the past where the std::sort implementation
// would proceed to do OOB reads. (rdar://106897934).
// Other algorithms like std::stable_sort, std::sort_heap do not go out of bounds but can produce
// incorrect results, we also want to assert on that.
// Sometimes std::sort does not go out of bounds as well, for example, right now if transitivity
// of equivalence is not met, std::sort can only produce incorrect result but would not fail.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <ranges>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "bad_comparator_values.h"
#include "check_assertion.h"
#include "invalid_comparator_utilities.h"

void check_oob_sort_read() {
  SortingFixture fixture(SORT_DATA);

  // Check the classic sorting algorithms
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(
        std::sort(copy.begin(), copy.end(), fixture.checked_predicate()),
        "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
  }
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(std::stable_sort(copy.begin(), copy.end(), fixture.checked_predicate()),
                               "Comparator does not induce a strict weak ordering");
  }
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(std::make_heap(copy.begin(), copy.end(), fixture.checked_predicate()),
                               "Comparator does not induce a strict weak ordering");
    TEST_LIBCPP_ASSERT_FAILURE(std::sort_heap(copy.begin(), copy.end(), fixture.checked_predicate()),
                               "Comparator does not induce a strict weak ordering");
  }
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(std::partial_sort(copy.begin(), copy.end(), copy.end(), fixture.checked_predicate()),
                               "Comparator does not induce a strict weak ordering");
  }
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    std::vector<std::size_t*> results(copy.size(), nullptr);
    TEST_LIBCPP_ASSERT_FAILURE(
        std::partial_sort_copy(copy.begin(), copy.end(), results.begin(), results.end(), fixture.checked_predicate()),
        "Comparator does not induce a strict weak ordering");
  }

  // Check the Ranges sorting algorithms
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(
        std::ranges::sort(copy, fixture.checked_predicate()),
        "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
  }
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::stable_sort(copy, fixture.checked_predicate()),
                               "Comparator does not induce a strict weak ordering");
  }
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(
        std::ranges::make_heap(copy, fixture.checked_predicate()), "Comparator does not induce a strict weak ordering");
    TEST_LIBCPP_ASSERT_FAILURE(
        std::ranges::sort_heap(copy, fixture.checked_predicate()), "Comparator does not induce a strict weak ordering");
  }
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::partial_sort(copy, copy.end(), fixture.checked_predicate()),
                               "Comparator does not induce a strict weak ordering");
  }
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    std::vector<std::size_t*> results(copy.size(), nullptr);
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::partial_sort_copy(copy, results, fixture.checked_predicate()),
                               "Comparator does not induce a strict weak ordering");
  }
}

void check_oob_nth_element_read() {
  SortingFixture fixture(NTH_ELEMENT_DATA);

  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(std::nth_element(copy.begin(), copy.begin(), copy.end(), fixture.checked_predicate()),
                               "Comparator does not induce a strict weak ordering");
  }

  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::nth_element(copy, copy.begin(), fixture.checked_predicate()),
                               "Comparator does not induce a strict weak ordering");
  }
}

struct FloatContainer {
  float value;
  bool operator<(const FloatContainer& other) const { return value < other.value; }
};

// Nans in floats do not satisfy strict weak ordering by breaking transitivity of equivalence.
std::vector<FloatContainer> generate_float_data() {
  std::vector<FloatContainer> floats(50);
  for (int i = 0; i < 50; ++i) {
    floats[i].value = static_cast<float>(i);
  }
  floats.push_back(FloatContainer{std::numeric_limits<float>::quiet_NaN()});
  std::shuffle(floats.begin(), floats.end(), std::default_random_engine());
  return floats;
}

void check_nan_floats() {
  auto floats = generate_float_data();
  TEST_LIBCPP_ASSERT_FAILURE(
      std::sort(floats.begin(), floats.end()), "Your comparator is not a valid strict-weak ordering");
  floats = generate_float_data();
  TEST_LIBCPP_ASSERT_FAILURE(
      std::stable_sort(floats.begin(), floats.end()), "Your comparator is not a valid strict-weak ordering");
  floats = generate_float_data();
  std::make_heap(floats.begin(), floats.end());
  TEST_LIBCPP_ASSERT_FAILURE(
      std::sort_heap(floats.begin(), floats.end()), "Your comparator is not a valid strict-weak ordering");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::ranges::sort(generate_float_data(), std::less()), "Your comparator is not a valid strict-weak ordering");
  TEST_LIBCPP_ASSERT_FAILURE(std::ranges::stable_sort(generate_float_data(), std::less()),
                             "Your comparator is not a valid strict-weak ordering");
  floats = generate_float_data();
  std::ranges::make_heap(floats, std::less());
  TEST_LIBCPP_ASSERT_FAILURE(
      std::ranges::sort_heap(floats, std::less()), "Your comparator is not a valid strict-weak ordering");
}

void check_irreflexive() {
  std::vector<int> v(1);
  TEST_LIBCPP_ASSERT_FAILURE(
      std::sort(v.begin(), v.end(), std::greater_equal<int>()), "Your comparator is not a valid strict-weak ordering");
  TEST_LIBCPP_ASSERT_FAILURE(std::stable_sort(v.begin(), v.end(), std::greater_equal<int>()),
                             "Your comparator is not a valid strict-weak ordering");
  std::make_heap(v.begin(), v.end(), std::greater_equal<int>());
  TEST_LIBCPP_ASSERT_FAILURE(std::sort_heap(v.begin(), v.end(), std::greater_equal<int>()),
                             "Comparator does not induce a strict weak ordering");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::ranges::sort(v, std::greater_equal<int>()), "Your comparator is not a valid strict-weak ordering");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::ranges::stable_sort(v, std::greater_equal<int>()), "Your comparator is not a valid strict-weak ordering");
  std::ranges::make_heap(v, std::greater_equal<int>());
  TEST_LIBCPP_ASSERT_FAILURE(
      std::ranges::sort_heap(v, std::greater_equal<int>()), "Comparator does not induce a strict weak ordering");
}

int main(int, char**) {
  check_oob_sort_read();

  check_oob_nth_element_read();

  check_nan_floats();

  check_irreflexive();

  return 0;
}
