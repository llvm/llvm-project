//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_set

// unordered_set(const unordered_set& u, const allocator_type& a);

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <unordered_set>

#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class Alloc>
void test_alloc(const Alloc& new_alloc) {
  { // Simple check
    using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

    int arr[] = {1, 2, 3};
    const Set orig(std::begin(arr), std::end(arr));
    Set copy(orig, new_alloc);
    LIBCPP_ASSERT(copy.bucket_count() == 5);
    assert(copy.size() == 3);
    assert(copy.count(1) == 1);
    assert(copy.count(2) == 1);
    assert(copy.count(3) == 1);
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(std::fabs(copy.load_factor() - static_cast<float>(copy.size()) / copy.bucket_count()) < FLT_EPSILON);
    assert(copy.max_load_factor() == 1.f);
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 5);
    assert(orig.size() == 3);
    assert(orig.count(1) == 1);
    assert(orig.count(2) == 1);
    assert(orig.count(3) == 1);
  }
  { // single element check
    using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

    int arr[] = {1};
    const Set orig(std::begin(arr), std::end(arr));
    Set copy(orig, new_alloc);
    LIBCPP_ASSERT(copy.bucket_count() == 2);
    assert(copy.size() == 1);
    assert(copy.count(1) == 1);
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(std::fabs(copy.load_factor() - static_cast<float>(copy.size()) / copy.bucket_count()) < FLT_EPSILON);
    assert(copy.max_load_factor() == 1.f);
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 2);
    assert(orig.size() == 1);
    assert(orig.count(1) == 1);
  }
  { // Copy empty map
    using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

    const Set orig;
    Set copy(orig, new_alloc);
    LIBCPP_ASSERT(copy.bucket_count() == 0);
    assert(copy.size() == 0);
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(copy.max_load_factor() == 1.f);
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 0);
    assert(orig.size() == 0);
  }
  { // Ensure that the hash function is copied
    using Set = std::unordered_set<int, test_hash<int>, std::equal_to<int>, Alloc>;
    const Set orig(0, test_hash<int>(23));
    Set copy(orig, new_alloc);
    assert(copy.hash_function() == test_hash<int>(23));
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    assert(orig.hash_function() == test_hash<int>(23));
  }
  { // Ensure that the quality comparator is copied
    using Set = std::unordered_set<int, std::hash<int>, test_equal_to<int>, Alloc>;
    const Set orig(0, std::hash<int>(), test_equal_to<int>(56));
    Set copy(orig, new_alloc);
    assert(copy.key_eq() == test_equal_to<int>(56));
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    assert(orig.key_eq() == test_equal_to<int>(56));
  }
}

void test() {
  test_alloc(std::allocator<int>());
  test_alloc(min_allocator<int>());
  test_alloc(test_allocator<int>(25));
}

int main(int, char**) {
  test();

  return 0;
}
