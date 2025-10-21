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
// class unordered_multimap

// unordered_multimap(const unordered_multimap& u);

#include <unordered_map>
#include <string>
#include <set>
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
    using V   = std::pair<const int, int>;
    using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

    V arr[] = {V(1, 2), V(2, 4), V(3, 1), V(3, 2)};
    const Map orig(std::begin(arr), std::end(arr));
    Map copy = orig;
    LIBCPP_ASSERT(copy.bucket_count() == 5);
    assert(copy.size() == 4);
    assert(copy.find(1)->second == 2);
    assert(copy.find(2)->second == 4);
    {
      auto range          = copy.equal_range(3);
      auto first_element  = std::next(range.first, 0);
      auto second_element = std::next(range.first, 1);
      auto end            = std::next(range.first, 2);

      assert(range.second == end);

      assert(first_element->second == 1 || first_element->second == 2);
      assert(second_element->second == 1 || second_element->second == 2);
      assert(second_element->second != range.first->second);
    }
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(std::fabs(copy.load_factor() - static_cast<float>(copy.size()) / copy.bucket_count()) < FLT_EPSILON);
    assert(copy.max_load_factor() == 1.f);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 5);
    assert(orig.size() == 4);
    assert(orig.find(1)->second == 2);
    assert(orig.find(2)->second == 4);
    {
      auto range          = orig.equal_range(3);
      auto first_element  = std::next(range.first, 0);
      auto second_element = std::next(range.first, 1);
      auto end            = std::next(range.first, 2);

      assert(range.second == end);

      assert(first_element->second == 1 || first_element->second == 2);
      assert(second_element->second == 1 || second_element->second == 2);
      assert(second_element->second != range.first->second);
    }
  }
  { // single element copy
    using V   = std::pair<const int, int>;
    using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

    V arr[] = {V(1, 2)};
    const Map orig(std::begin(arr), std::end(arr));
    Map copy = orig;
    LIBCPP_ASSERT(copy.bucket_count() == 2);
    assert(copy.size() == 1);
    assert(copy.find(1)->second == 2);
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(std::fabs(copy.load_factor() - static_cast<float>(copy.size()) / copy.bucket_count()) < FLT_EPSILON);
    assert(copy.max_load_factor() == 1.f);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 2);
    assert(orig.size() == 1);
    assert(orig.find(1)->second == 2);
  }
  { // Copy empty map
    using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

    const Map orig;
    Map copy = orig;
    LIBCPP_ASSERT(copy.bucket_count() == 0);
    assert(copy.size() == 0);
    assert(static_cast<std::size_t>(std::distance(copy.begin(), copy.end())) == copy.size());
    assert(copy.max_load_factor() == 1.f);

    // Check that orig is still what is expected
    LIBCPP_ASSERT(orig.bucket_count() == 0);
    assert(orig.size() == 0);
  }
  { // Ensure that the hash function is copied
    using Map = std::unordered_multimap<int, int, test_hash<int>, std::equal_to<int>, Alloc>;
    const Map orig(0, test_hash<int>(23));
    Map copy = orig;
    assert(copy.hash_function() == test_hash<int>(23));

    // Check that orig is still what is expected
    assert(orig.hash_function() == test_hash<int>(23));
  }
  { // Ensure that the quality comparator is copied
    using Map = std::unordered_multimap<int, int, std::hash<int>, test_equal_to<int>, Alloc>;
    const Map orig(0, std::hash<int>(), test_equal_to<int>(56));
    Map copy = orig;
    assert(copy.key_eq() == test_equal_to<int>(56));

    // Check that orig is still what is expected
    assert(orig.key_eq() == test_equal_to<int>(56));
  }
}

void test() {
  test_alloc<std::allocator<std::pair<const int, int> > >();
  test_alloc<min_allocator<std::pair<const int, int> > >();

  { // Ensure that the allocator is copied
    using V   = std::pair<const int, int>;
    using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, test_allocator<V> >;

    V arr[] = {V(1, 1), V(2, 3), V(3, 6)};
    const Map orig(begin(arr), end(arr), 0, std::hash<int>(), std::equal_to<int>(), test_allocator<V>(10));
    Map copy = orig;
    assert(copy.size() == 3);
    assert(copy.get_allocator() == test_allocator<V>(10));

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(orig.get_allocator() == test_allocator<V>(10));
    assert(orig.get_allocator().get_id() != test_alloc_base::moved_value);
  }

  { // Ensure that soccc is handled properly
    using V   = std::pair<const int, int>;
    using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, other_allocator<V> >;

    V arr[] = {V(1, 1), V(2, 3), V(3, 6)};
    const Map orig(begin(arr), end(arr), 0, std::hash<int>(), std::equal_to<int>(), other_allocator<V>(10));
    Map copy = orig;
    assert(copy.size() == 3);
    assert(copy.get_allocator() == other_allocator<V>(-2));

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(orig.get_allocator() == other_allocator<V>(10));
  }
}

int main(int, char**) {
  test();

  return 0;
}
