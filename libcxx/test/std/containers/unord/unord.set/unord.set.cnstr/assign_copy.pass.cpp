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

// unordered_set& operator=(const unordered_set& u);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <unordered_set>
#include <utility>
#include <vector>

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
      using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

      int arr[] = {1, 2, 4, 5};
      const Set orig(std::begin(arr), std::end(arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      Set copy(lhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 5);
      assert(copy.size() == 4);
      assert(copy.count(1) == 1);
      assert(copy.count(2) == 1);
      assert(copy.count(4) == 1);
      assert(copy.count(5) == 1);
      assert(std::next(copy.begin(), 4) == copy.end());

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 5);
      assert(orig.size() == 4);
      assert(orig.count(1) == 1);
      assert(orig.count(2) == 1);
      assert(orig.count(4) == 1);
      assert(orig.count(5) == 1);
      assert(std::next(orig.begin(), 4) == orig.end());
    }
    check_alloc_invariant();
    { // assign from an empty container into an empty one
      using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

      const Set orig(rhs_alloc);
      Set copy(lhs_alloc);
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
      using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

      int arr[] = {1, 2, 4, 5};
      const Set orig(rhs_alloc);
      Set copy(std::begin(arr), std::end(arr), 0, std::hash<int>(), std::equal_to<int>(), lhs_alloc);
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
      using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

      int arr[] = {1};
      const Set orig(std::begin(arr), std::end(arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      Set copy(lhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 2);
      assert(copy.size() == 1);
      assert(copy.count(1) == 1);

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 2);
      assert(orig.size() == 1);
      assert(orig.count(1) == 1);
    }
    { // assign from an empty container into a single-element one
      using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

      int arr[] = {1};
      const Set orig(rhs_alloc);
      Set copy(std::begin(arr), std::end(arr), 0, std::hash<int>(), std::equal_to<int>(), lhs_alloc);
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
      using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

      int arr[] = {1, 2, 4, 5};
      Set orig(std::begin(arr), std::end(arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      orig = static_cast<const Set&>(orig);
      LIBCPP_ASSERT(orig.bucket_count() == 5);
      assert(orig.size() == 4);
      assert(orig.count(1) == 1);
      assert(orig.count(2) == 1);
      assert(orig.count(4) == 1);
      assert(orig.count(5) == 1);
      assert(std::next(orig.begin(), 4) == orig.end());
    }
    check_alloc_invariant();
    { // with an empty map
      using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

      Set orig(rhs_alloc);
      orig = static_cast<const Set&>(orig);
      LIBCPP_ASSERT(orig.bucket_count() == 0);
      assert(orig.size() == 0);
      assert(orig.begin() == orig.end());
    }
    check_alloc_invariant();
  }
  {   // check assignment into a non-empty map
    { // LHS already contains elements, but fewer than the RHS
      using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

      int lhs_arr[] = {1, 2, 4, 5};
      const Set orig(std::begin(lhs_arr), std::end(lhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);

      int rhs_arr[] = {10, 13};
      Set copy(std::begin(rhs_arr), std::end(rhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 5);
      assert(copy.size() == 4);
      assert(copy.count(1) == 1);
      assert(copy.count(2) == 1);
      assert(copy.count(4) == 1);
      assert(copy.count(5) == 1);
      assert(std::next(copy.begin(), 4) == copy.end());

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 5);
      assert(orig.size() == 4);
      assert(orig.count(1) == 1);
      assert(orig.count(2) == 1);
      assert(orig.count(4) == 1);
      assert(orig.count(5) == 1);
      assert(std::next(orig.begin(), 4) == orig.end());
    }
    check_alloc_invariant();
    { // LHS contains the same number of elements as the RHS
      using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

      int lhs_arr[] = {1, 2, 4, 5};
      const Set orig(std::begin(lhs_arr), std::end(lhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);

      int rhs_arr[] = {10, 13, 12, 0};
      Set copy(std::begin(rhs_arr), std::end(rhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 5);
      assert(copy.size() == 4);
      assert(copy.count(1) == 1);
      assert(copy.count(2) == 1);
      assert(copy.count(4) == 1);
      assert(copy.count(5) == 1);
      assert(std::next(copy.begin(), 4) == copy.end());

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 5);
      assert(orig.size() == 4);
      assert(orig.count(1) == 1);
      assert(orig.count(2) == 1);
      assert(orig.count(4) == 1);
      assert(orig.count(5) == 1);
      assert(std::next(orig.begin(), 4) == orig.end());
    }
    check_alloc_invariant();
    { // LHS already contains more elements than the RHS
      using Set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

      int lhs_arr[] = {1, 2, 4, 5};
      const Set orig(std::begin(lhs_arr), std::end(lhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);

      int rhs_arr[] = {10, 13, 12, 0, 50, 2};
      Set copy(std::begin(rhs_arr), std::end(rhs_arr), 0, std::hash<int>(), std::equal_to<int>(), rhs_alloc);
      copy = orig;
      LIBCPP_ASSERT(copy.bucket_count() == 5);
      assert(copy.size() == 4);
      assert(copy.count(1) == 1);
      assert(copy.count(2) == 1);
      assert(copy.count(4) == 1);
      assert(copy.count(5) == 1);
      assert(std::next(copy.begin(), 4) == copy.end());

      // Check that orig is still what is expected
      LIBCPP_ASSERT(orig.bucket_count() == 5);
      assert(orig.size() == 4);
      assert(orig.count(1) == 1);
      assert(orig.count(2) == 1);
      assert(orig.count(4) == 1);
      assert(orig.count(5) == 1);
      assert(std::next(orig.begin(), 4) == orig.end());
    }
    check_alloc_invariant();
  }
}

void test() {
  test_alloc<std::allocator<int> >();
  test_alloc<min_allocator<int> >();

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
    test_alloc<tracking_allocator<int> >(lhs_allocs, rhs_allocs, AssertEmpty(lhs_allocs, rhs_allocs));
  }

  {   // Ensure that the hasher is copied
    { // when the container is non-empty
      using Set = std::unordered_set<int, test_hash<int> >;

      int arr[] = {1, 2, 3};
      const Set orig(std::begin(arr), std::end(arr), 0, test_hash<int>(5));
      Set copy;
      copy = orig;
      assert(copy.size() == 3);
      assert(copy.hash_function() == test_hash<int>(5));

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(orig.hash_function() == test_hash<int>(5));
    }
    { // when the container is empty
      using Set = std::unordered_set<int, test_hash<int> >;

      const Set orig(0, test_hash<int>(5));
      Set copy;
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
      using Set = std::unordered_set<int, std::hash<int>, test_equal_to<int> >;

      int arr[] = {1, 2, 3};
      const Set orig(std::begin(arr), std::end(arr), 0, std::hash<int>(), test_equal_to<int>(23));
      Set copy;
      copy = orig;
      assert(copy.size() == 3);
      assert(copy.key_eq() == test_equal_to<int>(23));

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(copy.key_eq() == test_equal_to<int>(23));
    }
    { // when the container is empty
      using Set = std::unordered_set<int, std::hash<int>, test_equal_to<int> >;

      const Set orig(0, std::hash<int>(), test_equal_to<int>(23));
      Set copy;
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
      using Set = std::unordered_set<int>;

      int arr[] = {1, 2, 3};
      Set orig(std::begin(arr), std::end(arr));
      orig.max_load_factor(33.f);
      Set copy;
      copy = orig;
      assert(copy.size() == 3);
      assert(copy.max_load_factor() == 33.f);

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(orig.max_load_factor() == 33.f);
    }
    { // when the container is empty
      using Set = std::unordered_set<int>;

      Set orig;
      orig.max_load_factor(17.f);
      Set copy;
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
        using Alloc = other_allocator<int>;
        using Set   = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

        int arr[] = {1, 2, 3};
        const Set orig(std::begin(arr), std::end(arr), 0, std::hash<int>(), std::equal_to<int>(), Alloc(3));
        Set copy(Alloc(1));
        copy = orig;
        assert(copy.get_allocator() == Alloc(3));
      }
      { // when the container is empty
        using Alloc = other_allocator<int>;
        using Set   = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

        const Set orig(Alloc(3));
        Set copy(Alloc(1));
        copy = orig;
        assert(copy.get_allocator() == Alloc(3));
      }
    }
    {   // pocca == false_type
      { // when the container is non-empty
        using Alloc = test_allocator<int>;
        using Set   = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

        int arr[] = {1, 2, 3};
        const Set orig(std::begin(arr), std::end(arr), 0, std::hash<int>(), std::equal_to<int>(), Alloc(3));
        Set copy(Alloc(1));
        copy = orig;
        assert(copy.get_allocator() == Alloc(1));
      }
      { // when the container is empty
        using Alloc = test_allocator<int>;
        using Set   = std::unordered_set<int, std::hash<int>, std::equal_to<int>, Alloc>;

        const Set orig(Alloc(3));
        Set copy(Alloc(1));
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
