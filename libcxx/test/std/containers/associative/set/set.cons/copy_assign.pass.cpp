//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// set& operator=(const set& s);

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iterator>
#include <set>
#include <vector>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

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
  {   // Test empty/non-empty set combinations
    { // assign from a non-empty container into an empty one
      using Set = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1, 2, 3};
      const Set orig(std::begin(arr), std::end(arr), std::less<int>(), rhs_alloc);
      Set copy(lhs_alloc);
      copy = orig;
      assert(copy.size() == 3);
      assert(*std::next(copy.begin(), 0) == 1);
      assert(*std::next(copy.begin(), 1) == 2);
      assert(*std::next(copy.begin(), 2) == 3);
      assert(std::next(copy.begin(), 3) == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(*std::next(orig.begin(), 0) == 1);
      assert(*std::next(orig.begin(), 1) == 2);
      assert(*std::next(orig.begin(), 2) == 3);
      assert(std::next(orig.begin(), 3) == orig.end());
    }
    check_alloc_invariant();
    { // assign from an empty container into an empty one
      using Set = std::set<int, std::less<int>, Alloc>;

      const Set orig(rhs_alloc);
      Set copy(lhs_alloc);
      copy = orig;
      assert(copy.size() == 0);
      assert(copy.begin() == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 0);
      assert(orig.begin() == orig.end());
    }
    check_alloc_invariant();
    { // assign from an empty container into a non-empty one
      using Set = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1, 2, 3};
      const Set orig(rhs_alloc);
      Set copy(std::begin(arr), std::end(arr), std::less<int>(), rhs_alloc);
      copy = orig;
      assert(copy.size() == 0);
      assert(copy.begin() == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 0);
      assert(orig.begin() == orig.end());
    }
  }

  {   // Ensure that self-assignment works correctly
    { // with a non-empty set
      using Set = std::set<int, std::less<int>, Alloc>;

      int arr[] = {1, 2, 3};
      Set orig(std::begin(arr), std::end(arr), std::less<int>(), rhs_alloc);
      orig = static_cast<const Set&>(orig);

      assert(orig.size() == 3);
      assert(*std::next(orig.begin(), 0) == 1);
      assert(*std::next(orig.begin(), 1) == 2);
      assert(*std::next(orig.begin(), 2) == 3);
      assert(std::next(orig.begin(), 3) == orig.end());
    }
    { // with an empty set
      using Set = std::set<int, std::less<int>, Alloc>;

      Set orig(rhs_alloc);
      orig = static_cast<const Set&>(orig);

      assert(orig.size() == 0);
      assert(orig.begin() == orig.end());
    }
  }

  { // check assignment into a non-empty set
    check_alloc_invariant();
    { // LHS already contains elements, but fewer than the RHS
      using Set = std::set<int, std::less<int>, Alloc>;

      int lhs_arr[] = {1, 2, 3};
      const Set orig(std::begin(lhs_arr), std::end(lhs_arr), std::less<int>(), rhs_alloc);

      int rhs_arr[] = {4, 5};
      Set copy(std::begin(rhs_arr), std::end(rhs_arr), std::less<int>(), lhs_alloc);
      copy = orig;
      assert(copy.size() == 3);
      assert(*std::next(copy.begin(), 0) == 1);
      assert(*std::next(copy.begin(), 1) == 2);
      assert(*std::next(copy.begin(), 2) == 3);
      assert(std::next(copy.begin(), 3) == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(*std::next(orig.begin(), 0) == 1);
      assert(*std::next(orig.begin(), 1) == 2);
      assert(*std::next(orig.begin(), 2) == 3);
      assert(std::next(orig.begin(), 3) == orig.end());
    }
    check_alloc_invariant();
    { // LHS contains the same number of elements as the RHS
      using Set = std::set<int, std::less<int>, Alloc>;

      int lhs_arr[] = {1, 2, 3};
      const Set orig(std::begin(lhs_arr), std::end(lhs_arr), std::less<int>(), rhs_alloc);

      int rhs_arr[] = {4, 5, 6};
      Set copy(std::begin(rhs_arr), std::end(rhs_arr), std::less<int>(), lhs_alloc);
      copy = orig;
      assert(copy.size() == 3);
      assert(*std::next(copy.begin(), 0) == 1);
      assert(*std::next(copy.begin(), 1) == 2);
      assert(*std::next(copy.begin(), 2) == 3);
      assert(std::next(copy.begin(), 3) == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(*std::next(orig.begin(), 0) == 1);
      assert(*std::next(orig.begin(), 1) == 2);
      assert(*std::next(orig.begin(), 2) == 3);
      assert(std::next(orig.begin(), 3) == orig.end());
    }
    check_alloc_invariant();
    { // LHS already contains more elements than the RHS
      using Set = std::set<int, std::less<int>, Alloc>;

      int lhs_arr[] = {1, 2, 3};
      const Set orig(std::begin(lhs_arr), std::end(lhs_arr), std::less<int>(), rhs_alloc);

      int rhs_arr[] = {4, 5, 6};
      Set copy(std::begin(rhs_arr), std::end(rhs_arr), std::less<int>(), lhs_alloc);
      copy = orig;
      assert(copy.size() == 3);
      assert(*std::next(copy.begin(), 0) == 1);
      assert(*std::next(copy.begin(), 1) == 2);
      assert(*std::next(copy.begin(), 2) == 3);
      assert(std::next(copy.begin(), 3) == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(*std::next(orig.begin(), 0) == 1);
      assert(*std::next(orig.begin(), 1) == 2);
      assert(*std::next(orig.begin(), 2) == 3);
      assert(std::next(orig.begin(), 3) == orig.end());
    }
    check_alloc_invariant();
    { // Make a somewhat larger set to exercise the algorithm a bit
      using Set = std::set<int, std::less<int>, Alloc>;

      Set orig(rhs_alloc);
      for (int i = 0; i != 50; ++i)
        orig.insert(i);

      Set copy(lhs_alloc);
      copy  = orig;
      int i = 0;
      for (auto v : copy) {
        assert(v == i++);
      }
    }
    check_alloc_invariant();
  }
}

void test() {
  test_alloc<std::allocator<int> >();
#if TEST_STD_VER >= 11
  test_alloc<min_allocator<int> >();

  { // Make sure we're allocating/deallocating nodes with the correct allocator
    // See https://llvm.org/PR29001
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
#endif

  { // Ensure that the comparator is copied
    int arr[] = {1, 2, 3};
    const std::set<int, test_less<int> > orig(std::begin(arr), std::end(arr), test_less<int>(3));
    std::set<int, test_less<int> > copy;
    copy = orig;
    assert(copy.size() == 3);
    assert(copy.key_comp() == test_less<int>(3));

    // Check that orig is still what is expected
    assert(orig.size() == 3);
    assert(orig.key_comp() == test_less<int>(3));
  }
}

int main(int, char**) {
  test();

  return 0;
}
