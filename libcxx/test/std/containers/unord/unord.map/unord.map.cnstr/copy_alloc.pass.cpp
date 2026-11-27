//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// unordered_map(const unordered_map& u, const allocator_type& a);

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <unordered_map>

#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class Alloc>
void test_alloc(const Alloc& new_alloc) {
  { // Simple check
    using V   = std::pair<const int, int>;
    using Map = std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

    V arr[] = {V(1, 2), V(2, 4), V(3, 1)};
    const Map orig(std::begin(arr), std::end(arr));
    Map copy(orig, new_alloc);
    LIBCPP_ASSERT(copy.bucket_count() == 5);
    assert(copy.size() == 3);
    assert(copy.at(1) == 2);
    assert(copy.at(2) == 4);
    assert(copy.at(3) == 1);
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(std::fabs(copy.load_factor() - static_cast<float>(copy.size()) / copy.bucket_count()) < FLT_EPSILON);
    assert(copy.max_load_factor() == 1.f);
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 5);
    assert(orig.size() == 3);
    assert(orig.at(1) == 2);
    assert(orig.at(2) == 4);
    assert(orig.at(3) == 1);
  }
  { // single element check
    using V   = std::pair<const int, int>;
    using Map = std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

    V arr[] = {V(1, 2)};
    const Map orig(std::begin(arr), std::end(arr));
    Map copy(orig, new_alloc);
    LIBCPP_ASSERT(copy.bucket_count() == 2);
    assert(copy.size() == 1);
    assert(copy.at(1) == 2);
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(std::fabs(copy.load_factor() - static_cast<float>(copy.size()) / copy.bucket_count()) < FLT_EPSILON);
    assert(copy.max_load_factor() == 1.f);
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 2);
    assert(orig.size() == 1);
    assert(orig.at(1) == 2);
  }
  { // Copy empty map
    using Map = std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

    const Map orig;
    Map copy(orig, new_alloc);
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
    using Map = std::unordered_map<int, int, test_hash<int>, std::equal_to<int>, Alloc>;
    const Map orig(0, test_hash<int>(23));
    Map copy(orig, new_alloc);
    assert(copy.hash_function() == test_hash<int>(23));
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    assert(orig.hash_function() == test_hash<int>(23));
  }
  { // Ensure that the quality comparator is copied
    using Map = std::unordered_map<int, int, std::hash<int>, test_equal_to<int>, Alloc>;
    const Map orig(0, std::hash<int>(), test_equal_to<int>(56));
    Map copy(orig, new_alloc);
    assert(copy.key_eq() == test_equal_to<int>(56));
    assert(copy.get_allocator() == new_alloc);

    // Check that orig is still what is expected
    assert(orig.key_eq() == test_equal_to<int>(56));
  }
}

void test() {
  test_alloc(std::allocator<std::pair<const int, int> >());
  test_alloc(min_allocator<std::pair<const int, int> >());
  test_alloc(test_allocator<std::pair<const int, int> >(25));
}

int main(int, char**) {
  test();

  return 0;
}
