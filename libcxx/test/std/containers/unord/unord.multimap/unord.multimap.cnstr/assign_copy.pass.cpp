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

// unordered_multimap& operator=(const unordered_multimap& u);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "../../../check_consecutive.h"
#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class T>
class tracking_allocator {
  std::vector<void*>* allocs_;

  template <class U>
  friend class tracking_allocator;

public:
  using value_type                             = T;
  using propagate_on_container_copy_assignment = std::true_type;

  tracking_allocator(std::vector<void*>& allocs) : allocs_(&allocs) {}

  template <class U>
  tracking_allocator(const tracking_allocator<U>& other) : allocs_(other.allocs_) {}

  T* allocate(std::size_t n) {
    T* allocation = std::allocator<T>().allocate(n);
    allocs_->push_back(allocation);
    return allocation;
  }

  void deallocate(T* ptr, std::size_t n) TEST_NOEXCEPT {
    auto res = std::remove(allocs_->begin(), allocs_->end(), ptr);
    assert(res != allocs_->end() && "Trying to deallocate memory from different allocator?");
    allocs_->erase(res);
    std::allocator<T>().deallocate(ptr, n);
  }

  friend bool operator==(const tracking_allocator& lhs, const tracking_allocator& rhs) {
    return lhs.allocs_ == rhs.allocs_;
  }

  friend bool operator!=(const tracking_allocator& lhs, const tracking_allocator& rhs) {
    return lhs.allocs_ != rhs.allocs_;
  }
};

struct NoOp {
  void operator()() {}
};

template <class Alloc, class AllocatorInvariant = NoOp>
void test_alloc(const Alloc& lhs_alloc                   = Alloc(),
                const Alloc& rhs_alloc                   = Alloc(),
                AllocatorInvariant check_alloc_invariant = NoOp()) {
  {   // Test empty/non-empty combinations
    { // assign from a non-empty container into an empty one
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

      V arr[] = {V(1, 1), V(2, 3), V(4, 4), V(4, 2)};
      const Map orig(begin(arr), end(arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      Map copy(lhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 5);
      assert(copy.size() == 4);
      assert(copy.find(1)->second == 1);
      assert(copy.find(2)->second == 3);
      {
        auto range          = copy.equal_range(4);
        auto first_element  = std::next(range.first, 0);
        auto second_element = std::next(range.first, 1);
        auto end            = std::next(range.first, 2);

        assert(range.second == end);

        assert(first_element->second == 2 || first_element->second == 4);
        assert(second_element->second == 2 || second_element->second == 4);
        assert(second_element->second != range.first->second);
      }
      assert(std::next(copy.begin(), 4) == copy.end());

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 5);
      assert(orig.size() == 4);
      assert(orig.find(1)->second == 1);
      assert(orig.find(2)->second == 3);
      {
        auto range          = orig.equal_range(4);
        auto first_element  = std::next(range.first, 0);
        auto second_element = std::next(range.first, 1);
        auto end            = std::next(range.first, 2);

        assert(range.second == end);

        assert(first_element->second == 2 || first_element->second == 4);
        assert(second_element->second == 2 || second_element->second == 4);
        assert(second_element->second != range.first->second);
      }
      assert(std::next(orig.begin(), 4) == orig.end());
    }
    check_alloc_invariant();
    { // assign from an empty container into an empty one
      using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

      const Map orig(rhs_alloc);
      Map copy(lhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 0);
      assert(copy.size() == 0);
      assert(copy.begin() == copy.end());

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 0);
      assert(orig.size() == 0);
      assert(orig.begin() == orig.end());
    }
    check_alloc_invariant();
    { // assign from an empty container into a non-empty one
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

      V arr[] = {V(1, 1), V(2, 3), V(4, 4), V(5, 2)};
      const Map orig(rhs_alloc);
      Map copy(begin(arr), end(arr), 0, std::hash<int>(), std::equal_to<int>(), lhs_alloc);
      copy = orig;
      // Depending on whether the allocator is propagated the bucked count can change
      LIBCPP_ASSERT(copy.bucket_count() == 5 || copy.bucket_count() == 0);
      assert(copy.size() == 0);
      assert(copy.begin() == copy.end());

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 0);
      assert(orig.size() == 0);
      assert(orig.begin() == orig.end());
    }
    check_alloc_invariant();
  }
  {   // Test empty/one-element copies. In our implementation that's a special case.
    { // assign from a single-element container into an empty one
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

      V arr[] = {V(1, 1)};
      const Map orig(begin(arr), end(arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      Map copy(lhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 2);
      assert(copy.size() == 1);
      assert(copy.find(1)->second == 1);

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 2);
      assert(orig.size() == 1);
      assert(orig.find(1)->second == 1);
    }
    { // assign from an empty container into a single-element one
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

      V arr[] = {V(1, 1)};
      const Map orig(rhs_alloc);
      Map copy(begin(arr), end(arr), 0, std::hash<int>(), std::equal_to<int>(), lhs_alloc);
      copy = orig;
      // Depending on whether the allocator is propagated the bucked count can change
      LIBCPP_ASSERT(copy.bucket_count() == 2 || copy.bucket_count() == 0);
      assert(copy.size() == 0);

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 0);
      assert(orig.size() == 0);
    }
  }
  {   // Ensure that self-assignment works correctly
    { // with a non-empty map
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

      V arr[] = {V(1, 1), V(2, 3), V(4, 4), V(5, 2)};
      Map orig(begin(arr), end(arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      orig = static_cast<const Map&>(orig);
      LIBCPP_ASSERT(orig.bucket_count() == 5);
      assert(orig.size() == 4);
      assert(orig.find(1)->second == 1);
      assert(orig.find(2)->second == 3);
      assert(orig.find(4)->second == 4);
      assert(orig.find(5)->second == 2);
      assert(std::next(orig.begin(), 4) == orig.end());
    }
    check_alloc_invariant();
    { // with an empty map
      using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

      Map orig(rhs_alloc);
      orig = static_cast<const Map&>(orig);
      LIBCPP_ASSERT(orig.bucket_count() == 0);
      assert(orig.size() == 0);
      assert(orig.begin() == orig.end());
    }
    check_alloc_invariant();
  }
  {   // check assignment into a non-empty map
    { // LHS already contains elements, but fewer than the RHS
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

      V lhs_arr[] = {V(1, 1), V(2, 3), V(4, 4), V(5, 2)};
      const Map orig(begin(lhs_arr), end(lhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);

      V rhs_arr[] = {V(10, 4), V(13, 5)};
      Map copy(begin(rhs_arr), end(rhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 5);
      assert(copy.size() == 4);
      assert(copy.find(1)->second == 1);
      assert(copy.find(2)->second == 3);
      assert(copy.find(4)->second == 4);
      assert(copy.find(5)->second == 2);
      assert(std::next(copy.begin(), 4) == copy.end());

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 5);
      assert(orig.size() == 4);
      assert(orig.find(1)->second == 1);
      assert(orig.find(2)->second == 3);
      assert(orig.find(4)->second == 4);
      assert(orig.find(5)->second == 2);
      assert(std::next(orig.begin(), 4) == orig.end());
    }
    check_alloc_invariant();
    { // LHS contains the same number of elements as the RHS
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

      V lhs_arr[] = {V(1, 1), V(2, 3), V(4, 4), V(5, 2)};
      const Map orig(begin(lhs_arr), end(lhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);

      V rhs_arr[] = {V(10, 4), V(13, 5), V(12, 324), V(0, 54)};
      Map copy(begin(rhs_arr), end(rhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 5);
      assert(copy.size() == 4);
      assert(copy.find(1)->second == 1);
      assert(copy.find(2)->second == 3);
      assert(copy.find(4)->second == 4);
      assert(copy.find(5)->second == 2);
      assert(std::next(copy.begin(), 4) == copy.end());

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 5);
      assert(orig.size() == 4);
      assert(orig.find(1)->second == 1);
      assert(orig.find(2)->second == 3);
      assert(orig.find(4)->second == 4);
      assert(orig.find(5)->second == 2);
      assert(std::next(orig.begin(), 4) == orig.end());
    }
    check_alloc_invariant();
    { // LHS already contains more elements than the RHS
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

      V lhs_arr[] = {V(1, 1), V(2, 3), V(4, 4), V(5, 2)};
      const Map orig(begin(lhs_arr), end(lhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);

      V rhs_arr[] = {V(10, 4), V(13, 5), V(12, 324), V(0, 54), V(50, 5), V(2, 5)};
      Map copy(begin(rhs_arr), end(rhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 5);
      assert(copy.size() == 4);
      assert(copy.find(1)->second == 1);
      assert(copy.find(2)->second == 3);
      assert(copy.find(4)->second == 4);
      assert(copy.find(5)->second == 2);
      assert(std::next(copy.begin(), 4) == copy.end());

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 5);
      assert(orig.size() == 4);
      assert(orig.find(1)->second == 1);
      assert(orig.find(2)->second == 3);
      assert(orig.find(4)->second == 4);
      assert(orig.find(5)->second == 2);
      assert(std::next(orig.begin(), 4) == orig.end());
    }
    check_alloc_invariant();
  }
}

void test() {
  test_alloc<std::allocator<std::pair<const int, int> > >();
  test_alloc<min_allocator<std::pair<const int, int> > >();

  { // Make sure we're allocating/deallocating nodes with the correct allocator
    // See https://llvm.org/PR29001 (report is for std::map, but the unordered containers have the same optimization)
    class AssertEmpty {
      std::vector<void*>* lhs_allocs_;
      std::vector<void*>* rhs_allocs_;

    public:
      AssertEmpty(std::vector<void*>& lhs_allocs, std::vector<void*>& rhs_allocs)
          : lhs_allocs_(&lhs_allocs), rhs_allocs_(&rhs_allocs) {}

      void operator()() {
        assert(lhs_allocs_->empty());
        assert(rhs_allocs_->empty());
      }
    };

    std::vector<void*> lhs_allocs;
    std::vector<void*> rhs_allocs;
    test_alloc<tracking_allocator<std::pair<const int, int> > >(
        lhs_allocs, rhs_allocs, AssertEmpty(lhs_allocs, rhs_allocs));
  }

  {   // Ensure that the hasher is copied
    { // when the container is non-empty
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int, test_hash<int> >;

      V arr[] = {V(1, 1), V(2, 2), V(3, 3)};
      const Map orig(begin(arr), end(arr), 0, test_hash<int>(5));
      Map copy;
      copy = orig;
      assert(copy.size() == 3);
      assert(copy.hash_function() == test_hash<int>(5));

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(orig.hash_function() == test_hash<int>(5));
    }
    { // when the container is empty
      using Map = std::unordered_multimap<int, int, test_hash<int> >;

      const Map orig(0, test_hash<int>(5));
      Map copy;
      copy = orig;
      assert(copy.empty());
      assert(copy.hash_function() == test_hash<int>(5));

      // Check that orig is still what is expected
      assert(orig.empty());
      assert(orig.hash_function() == test_hash<int>(5));
    }
  }

  {   // Ensure that the equality comparator is copied
    { // when the container is non-empty
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int, std::hash<int>, test_equal_to<int> >;

      V arr[] = {V(1, 1), V(2, 2), V(3, 3)};
      const Map orig(begin(arr), end(arr), 0, std::hash<int>(), test_equal_to<int>(23));
      Map copy;
      copy = orig;
      assert(copy.size() == 3);
      assert(copy.key_eq() == test_equal_to<int>(23));

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(copy.key_eq() == test_equal_to<int>(23));
    }
    { // when the container is empty
      using Map = std::unordered_multimap<int, int, std::hash<int>, test_equal_to<int> >;

      const Map orig(0, std::hash<int>(), test_equal_to<int>(23));
      Map copy;
      copy = orig;
      assert(copy.size() == 0);
      assert(copy.key_eq() == test_equal_to<int>(23));

      // Check that orig is still what is expected
      assert(orig.size() == 0);
      assert(copy.key_eq() == test_equal_to<int>(23));
    }
  }

  {   // Ensure that the max load factor is copied
    { // when the container is non-empty
      using V   = std::pair<const int, int>;
      using Map = std::unordered_multimap<int, int>;

      V arr[] = {V(1, 1), V(2, 2), V(3, 3)};
      Map orig(begin(arr), end(arr));
      orig.max_load_factor(33.f);
      Map copy;
      copy = orig;
      assert(copy.size() == 3);
      assert(copy.max_load_factor() == 33.f);

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(orig.max_load_factor() == 33.f);
    }
    { // when the container is empty
      using Map = std::unordered_multimap<int, int>;

      Map orig;
      orig.max_load_factor(17.f);
      Map copy;
      copy = orig;
      assert(copy.size() == 0);
      assert(copy.max_load_factor() == 17.f);

      // Check that orig is still what is expected
      assert(orig.size() == 0);
      assert(orig.max_load_factor() == 17.f);
    }
  }

  {     // Check that pocca is handled properly
    {   // pocca == true_type
      { // when the container is non-empty
        using V     = std::pair<const int, int>;
        using Alloc = other_allocator<V>;
        using Map   = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

        V arr[] = {V(1, 1), V(2, 2), V(3, 3)};
        const Map orig(begin(arr), end(arr), 0, std::hash<int>(), std::equal_to<int>(), Alloc(3));
        Map copy(Alloc(1));
        copy = orig;
        assert(copy.get_allocator() == Alloc(3));
      }
      { // when the container is empty
        using Alloc = other_allocator<std::pair<const int, int> >;
        using Map   = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

        const Map orig(Alloc(3));
        Map copy(Alloc(1));
        copy = orig;
        assert(copy.get_allocator() == Alloc(3));
      }
    }
    {   // pocca == false_type
      { // when the container is non-empty
        using V     = std::pair<const int, int>;
        using Alloc = test_allocator<V>;
        using Map   = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

        V arr[] = {V(1, 1), V(2, 2), V(3, 3)};
        const Map orig(begin(arr), end(arr), 0, std::hash<int>(), std::equal_to<int>(), Alloc(3));
        Map copy(Alloc(1));
        copy = orig;
        assert(copy.get_allocator() == Alloc(1));
      }
      { // when the container is empty
        using Alloc = test_allocator<std::pair<const int, int> >;
        using Map   = std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

        const Map orig(Alloc(3));
        Map copy(Alloc(1));
        copy = orig;
        assert(copy.get_allocator() == Alloc(1));
      }
    }
  }
}

int main(int, char**) {
  test();

  return 0;
}
