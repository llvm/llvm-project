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
// class unordered_multiset

// unordered_multiset(const unordered_multiset& u);

#include <unordered_set>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>

#include "test_macros.h"
#include "../../../check_consecutive.h"
#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class Alloc>
void test_alloc() {
  { // Simple check
    using Set = std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, Alloc>;

    int arr[] = {1, 2, 3, 3};
    const Set orig(std::begin(arr), std::end(arr));
    Set copy = orig;
    LIBCPP_ASSERT(copy.bucket_count() == 5);
    assert(copy.size() == 4);
    assert(copy.count(1) == 1);
    assert(copy.count(2) == 1);
    assert(copy.count(3) == 2);
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(std::fabs(copy.load_factor() - static_cast<float>(copy.size()) / copy.bucket_count()) < FLT_EPSILON);
    assert(copy.max_load_factor() == 1.f);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 5);
    assert(orig.size() == 4);
    assert(orig.count(1) == 1);
    assert(orig.count(2) == 1);
    assert(orig.count(3) == 2);
  }
  { // single element copy
    using Set = std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, Alloc>;

    int arr[] = {1};
    const Set orig(std::begin(arr), std::end(arr));
    Set copy = orig;
    LIBCPP_ASSERT(copy.bucket_count() == 2);
    assert(copy.size() == 1);
    assert(copy.count(1) == 1);
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(std::fabs(copy.load_factor() - static_cast<float>(copy.size()) / copy.bucket_count()) < FLT_EPSILON);
    assert(copy.max_load_factor() == 1.f);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 2);
    assert(orig.size() == 1);
    assert(orig.count(1) == 1);
  }
  { // Copy empty map
    using Set = std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, Alloc>;

    const Set orig;
    Set copy = orig;
    LIBCPP_ASSERT(copy.bucket_count() == 0);
    assert(copy.size() == 0);
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(copy.max_load_factor() == 1.f);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 0);
    assert(orig.size() == 0);
  }
  { // Ensure that the hash function is copied
    using Set = std::unordered_multiset<int, test_hash<int>, std::equal_to<int>, Alloc>;
    const Set orig(0, test_hash<int>(23));
    Set copy = orig;
    assert(copy.hash_function() == test_hash<int>(23));

    // Check that orig is still what is expected
    assert(orig.hash_function() == test_hash<int>(23));
  }
  { // Ensure that the quality comparator is copied
    using Set = std::unordered_multiset<int, std::hash<int>, test_equal_to<int>, Alloc>;
    const Set orig(0, std::hash<int>(), test_equal_to<int>(56));
    Set copy = orig;
    assert(copy.key_eq() == test_equal_to<int>(56));

    // Check that orig is still what is expected
    assert(orig.key_eq() == test_equal_to<int>(56));
  }
}

void test() {
  test_alloc<std::allocator<int> >();
  test_alloc<min_allocator<int> >();

  { // Ensure that the allocator is copied
    using Set = std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, test_allocator<int> >;

    int arr[] = {1, 2, 3};
    const Set orig(std::begin(arr), std::end(arr), 0, std::hash<int>(), std::equal_to<int>(), test_allocator<int>(10));
    Set copy = orig;
    assert(copy.size() == 3);
    assert(copy.get_allocator() == test_allocator<int>(10));

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(orig.get_allocator() == test_allocator<int>(10));
    assert(orig.get_allocator().get_id() != test_alloc_base::moved_value);
  }

  { // Ensure that soccc is handled properly
    using Set = std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, other_allocator<int> >;

    int arr[] = {1, 2, 3};
    const Set orig(std::begin(arr), std::end(arr), 0, std::hash<int>(), std::equal_to<int>(), other_allocator<int>(10));
    Set copy = orig;
    assert(copy.size() == 3);
    assert(copy.get_allocator() == other_allocator<int>(-2));

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(orig.get_allocator() == other_allocator<int>(10));
  }
}

int main(int, char**) {
  test();

  return 0;
}
