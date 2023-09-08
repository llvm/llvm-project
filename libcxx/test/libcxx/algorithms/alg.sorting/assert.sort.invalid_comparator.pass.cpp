//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: stdlib=libc++

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: !libcpp-hardening-mode=debug
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG_STRICT_WEAK_ORDERING_CHECK
// When the debug mode is enabled, this test fails because we actually catch on the fly that the comparator is not
// a strict-weak ordering before we catch that we'd dereference out-of-bounds inside std::sort, which leads to different
// errors than the ones tested below.
// XFAIL: libcpp-hardening-mode=debug

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

void check_oob_sort_read() {
    std::map<std::size_t, std::map<std::size_t, bool>> comparison_results; // terrible for performance, but really convenient
    for (auto line : std::views::split(DATA, '\n') | std::views::filter([](auto const& line) { return !line.empty(); })) {
        auto values = std::views::split(line, ' ');
        auto it = values.begin();
        std::size_t left = std::stol(std::string((*it).data(), (*it).size()));
        it = std::next(it);
        std::size_t right = std::stol(std::string((*it).data(), (*it).size()));
        it = std::next(it);
        bool result = static_cast<bool>(std::stol(std::string((*it).data(), (*it).size())));
        comparison_results[left][right] = result;
    }
    auto predicate = [&](std::size_t* left, std::size_t* right) {
        assert(left != nullptr && right != nullptr && "something is wrong with the test");
        assert(comparison_results.contains(*left) && comparison_results[*left].contains(*right) && "malformed input data?");
        return comparison_results[*left][*right];
    };

    std::vector<std::unique_ptr<std::size_t>> elements;
    std::set<std::size_t*> valid_ptrs;
    for (std::size_t i = 0; i != comparison_results.size(); ++i) {
        elements.push_back(std::make_unique<std::size_t>(i));
        valid_ptrs.insert(elements.back().get());
    }

    auto checked_predicate = [&](size_t* left, size_t* right) {
        // If the pointers passed to the comparator are not in the set of pointers we
        // set up above, then we're being passed garbage values from the algorithm
        // because we're reading OOB.
        assert(valid_ptrs.contains(left));
        assert(valid_ptrs.contains(right));
        return predicate(left, right);
    };

    // Check the classic sorting algorithms
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        TEST_LIBCPP_ASSERT_FAILURE(std::sort(copy.begin(), copy.end(), checked_predicate), "Would read out of bounds");
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        TEST_LIBCPP_ASSERT_FAILURE(std::stable_sort(copy.begin(), copy.end(), checked_predicate), "not a valid strict-weak ordering");
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::make_heap(copy.begin(), copy.end(), checked_predicate); // doesn't go OOB even with invalid comparator
        TEST_LIBCPP_ASSERT_FAILURE(std::sort_heap(copy.begin(), copy.end(), checked_predicate), "not a valid strict-weak ordering");
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        TEST_LIBCPP_ASSERT_FAILURE(std::partial_sort(copy.begin(), copy.end(), copy.end(), checked_predicate), "not a valid strict-weak ordering");
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::vector<std::size_t*> results(copy.size(), nullptr);
       TEST_LIBCPP_ASSERT_FAILURE(std::partial_sort_copy(copy.begin(), copy.end(), results.begin(), results.end(), checked_predicate), "not a valid strict-weak ordering");
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::nth_element(copy.begin(), copy.end(), copy.end(), checked_predicate); // doesn't go OOB even with invalid comparator
    }

    // Check the Ranges sorting algorithms
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        TEST_LIBCPP_ASSERT_FAILURE(std::ranges::sort(copy, checked_predicate), "Would read out of bounds");
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        TEST_LIBCPP_ASSERT_FAILURE(std::ranges::stable_sort(copy, checked_predicate), "not a valid strict-weak ordering");
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::ranges::make_heap(copy, checked_predicate); // doesn't go OOB even with invalid comparator
        TEST_LIBCPP_ASSERT_FAILURE(std::ranges::sort_heap(copy, checked_predicate), "not a valid strict-weak ordering");
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        TEST_LIBCPP_ASSERT_FAILURE(std::ranges::partial_sort(copy, copy.end(), checked_predicate), "not a valid strict-weak ordering");
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::vector<std::size_t*> results(copy.size(), nullptr);
        TEST_LIBCPP_ASSERT_FAILURE(std::ranges::partial_sort_copy(copy, results, checked_predicate), "not a valid strict-weak ordering");
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::ranges::nth_element(copy, copy.end(), checked_predicate); // doesn't go OOB even with invalid comparator
    }
}

struct FloatContainer {
  float value;
  bool operator<(const FloatContainer& other) const {
    return value < other.value;
  }
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
    TEST_LIBCPP_ASSERT_FAILURE(std::sort(floats.begin(), floats.end()), "not a valid strict-weak ordering");
    floats = generate_float_data();
    TEST_LIBCPP_ASSERT_FAILURE(std::stable_sort(floats.begin(), floats.end()), "not a valid strict-weak ordering");
    floats = generate_float_data();
    std::make_heap(floats.begin(), floats.end());
    TEST_LIBCPP_ASSERT_FAILURE(std::sort_heap(floats.begin(), floats.end()), "not a valid strict-weak ordering");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::sort(generate_float_data(), std::less()), "not a valid strict-weak ordering");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::stable_sort(generate_float_data(), std::less()), "not a valid strict-weak ordering");
    floats = generate_float_data();
    std::ranges::make_heap(floats, std::less());
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::sort_heap(floats, std::less()), "not a valid strict-weak ordering");
}

void check_irreflexive() {
    std::vector<int> v(1);
    TEST_LIBCPP_ASSERT_FAILURE(std::sort(v.begin(), v.end(), std::greater_equal<int>()), "not a valid strict-weak ordering");
    TEST_LIBCPP_ASSERT_FAILURE(std::stable_sort(v.begin(), v.end(), std::greater_equal<int>()), "not a valid strict-weak ordering");
    std::make_heap(v.begin(), v.end(), std::greater_equal<int>());
    TEST_LIBCPP_ASSERT_FAILURE(std::sort_heap(v.begin(), v.end(), std::greater_equal<int>()), "not a valid strict-weak ordering");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::sort(v, std::greater_equal<int>()), "not a valid strict-weak ordering");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::stable_sort(v, std::greater_equal<int>()), "not a valid strict-weak ordering");
    std::ranges::make_heap(v, std::greater_equal<int>());
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::sort_heap(v, std::greater_equal<int>()), "not a valid strict-weak ordering");
}

int main(int, char**) {

    check_oob_sort_read();

    check_nan_floats();

    check_irreflexive();

    return 0;
}
