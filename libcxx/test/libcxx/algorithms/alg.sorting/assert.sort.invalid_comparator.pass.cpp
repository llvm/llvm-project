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
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// This test uses a specific combination of an invalid comparator and sequence of values to
// ensure that our sorting functions do not go out-of-bounds in that case. Instead, we should
// fail loud with an assertion. The specific issue we're looking for here is when the comparator
// does not satisfy the following property:
//
//    comp(a, b) implies that !comp(b, a)
//
// In other words,
//
//    a < b implies that !(b < a)
//
// If this is not satisfied, we have seen issues in the past where the std::sort implementation
// would proceed to do OOB reads.

// When the debug mode is enabled, this test fails because we actually catch that the comparator
// is not a strict-weak ordering before we catch that we'd dereference out-of-bounds inside std::sort,
// which leads to different errors than the ones tested below.
// XFAIL: libcpp-has-debug-mode

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <map>
#include <memory>
#include <ranges>
#include <set>
#include <string>
#include <vector>

#include "check_assertion.h"

std::string DATA =
#   include "bad_comparator_values.dat"
;

int main(int, char**) {
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
        std::stable_sort(copy.begin(), copy.end(), checked_predicate); // doesn't go OOB even with invalid comparator
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::partial_sort(copy.begin(), copy.begin(), copy.end(), checked_predicate); // doesn't go OOB even with invalid comparator
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::vector<std::size_t*> results(copy.size(), nullptr);
        std::partial_sort_copy(copy.begin(), copy.end(), results.begin(), results.end(), checked_predicate); // doesn't go OOB even with invalid comparator
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
        std::ranges::stable_sort(copy, checked_predicate); // doesn't go OOB even with invalid comparator
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::ranges::partial_sort(copy, copy.begin(), checked_predicate); // doesn't go OOB even with invalid comparator
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::vector<std::size_t*> results(copy.size(), nullptr);
        std::ranges::partial_sort_copy(copy, results, checked_predicate); // doesn't go OOB even with invalid comparator
    }
    {
        std::vector<std::size_t*> copy;
        for (auto const& e : elements)
            copy.push_back(e.get());
        std::ranges::nth_element(copy, copy.end(), checked_predicate); // doesn't go OOB even with invalid comparator
    }

    return 0;
}
