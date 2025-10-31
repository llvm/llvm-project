//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// multimap& operator=(const multimap& m);

#include <algorithm>
#include <cassert>
#include <map>
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
  {   // Test empty/non-empty multimap combinations
    { // assign from a non-empty container into an empty one
      using V   = std::pair<const int, int>;
      using Map = std::multimap<int, int, std::less<int>, Alloc>;

      V arr[] = {V(1, 1), V(2, 3), V(2, 6)};
      const Map orig(begin(arr), end(arr), std::less<int>(), rhs_alloc);
      Map copy(lhs_alloc);
      copy = orig;
      assert(copy.size() == 3);
      assert(*std::next(copy.begin(), 0) == V(1, 1));
      assert(*std::next(copy.begin(), 1) == V(2, 3));
      assert(*std::next(copy.begin(), 2) == V(2, 6));
      assert(std::next(copy.begin(), 3) == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(*std::next(orig.begin(), 0) == V(1, 1));
      assert(*std::next(orig.begin(), 1) == V(2, 3));
      assert(*std::next(orig.begin(), 2) == V(2, 6));
      assert(std::next(orig.begin(), 3) == orig.end());
    }
    check_alloc_invariant();
    { // assign from an empty container into an empty one
      using Map = std::multimap<int, int, std::less<int>, Alloc>;

      const Map orig(rhs_alloc);
      Map copy(lhs_alloc);
      copy = orig;
      assert(copy.size() == 0);
      assert(copy.begin() == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 0);
      assert(orig.begin() == orig.end());
    }
    check_alloc_invariant();
    { // assign from an empty container into a non-empty one
      using V   = std::pair<const int, int>;
      using Map = std::multimap<int, int, std::less<int>, Alloc>;

      V arr[] = {V(1, 1), V(2, 3), V(2, 6)};
      const Map orig(rhs_alloc);
      Map copy(begin(arr), end(arr), std::less<int>(), rhs_alloc);
      copy = orig;
      assert(copy.size() == 0);
      assert(copy.begin() == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 0);
      assert(orig.begin() == orig.end());
    }
  }

  {   // Ensure that self-assignment works correctly
    { // with a non-empty multimap
      using V   = std::pair<const int, int>;
      using Map = std::multimap<int, int, std::less<int>, Alloc>;

      V arr[] = {V(1, 1), V(2, 3), V(2, 6)};
      Map orig(begin(arr), end(arr), std::less<int>(), rhs_alloc);
      orig = static_cast<const Map&>(orig);

      assert(orig.size() == 3);
      assert(*std::next(orig.begin(), 0) == V(1, 1));
      assert(*std::next(orig.begin(), 1) == V(2, 3));
      assert(*std::next(orig.begin(), 2) == V(2, 6));
      assert(std::next(orig.begin(), 3) == orig.end());
    }
    { // with an empty multimap
      using Map = std::multimap<int, int, std::less<int>, Alloc>;

      Map orig(rhs_alloc);
      orig = static_cast<const Map&>(orig);

      assert(orig.size() == 0);
      assert(orig.begin() == orig.end());
    }
  }

  { // check assignment into a non-empty multimap
    check_alloc_invariant();
    { // LHS already contains elements, but fewer than the RHS
      using V   = std::pair<const int, int>;
      using Map = std::multimap<int, int, std::less<int>, Alloc>;

      V lhs_arr[] = {V(1, 1), V(2, 3), V(2, 6)};
      const Map orig(begin(lhs_arr), end(lhs_arr), std::less<int>(), rhs_alloc);

      V rhs_arr[] = {V(4, 2), V(5, 1)};
      Map copy(begin(rhs_arr), end(rhs_arr), std::less<int>(), lhs_alloc);
      copy = orig;
      assert(copy.size() == 3);
      assert(*std::next(copy.begin(), 0) == V(1, 1));
      assert(*std::next(copy.begin(), 1) == V(2, 3));
      assert(*std::next(copy.begin(), 2) == V(2, 6));
      assert(std::next(copy.begin(), 3) == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(*std::next(orig.begin(), 0) == V(1, 1));
      assert(*std::next(orig.begin(), 1) == V(2, 3));
      assert(*std::next(orig.begin(), 2) == V(2, 6));
      assert(std::next(orig.begin(), 3) == orig.end());
    }
    check_alloc_invariant();
    { // LHS contains the same number of elements as the RHS
      using V   = std::pair<const int, int>;
      using Map = std::multimap<int, int, std::less<int>, Alloc>;

      V lhs_arr[] = {V(1, 1), V(2, 3), V(2, 6)};
      const Map orig(begin(lhs_arr), end(lhs_arr), std::less<int>(), rhs_alloc);

      V rhs_arr[] = {V(4, 2), V(5, 1), V(6, 0)};
      Map copy(begin(rhs_arr), end(rhs_arr), std::less<int>(), lhs_alloc);
      copy = orig;
      assert(copy.size() == 3);
      assert(*std::next(copy.begin(), 0) == V(1, 1));
      assert(*std::next(copy.begin(), 1) == V(2, 3));
      assert(*std::next(copy.begin(), 2) == V(2, 6));
      assert(std::next(copy.begin(), 3) == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(*std::next(orig.begin(), 0) == V(1, 1));
      assert(*std::next(orig.begin(), 1) == V(2, 3));
      assert(*std::next(orig.begin(), 2) == V(2, 6));
      assert(std::next(orig.begin(), 3) == orig.end());
    }
    check_alloc_invariant();
    { // LHS already contains more elements than the RHS
      using V   = std::pair<const int, int>;
      using Map = std::multimap<int, int, std::less<int>, Alloc>;

      V lhs_arr[] = {V(1, 1), V(2, 3), V(2, 6)};
      const Map orig(begin(lhs_arr), end(lhs_arr), std::less<int>(), rhs_alloc);

      V rhs_arr[] = {V(4, 2), V(5, 1), V(6, 0), V(7, 9)};
      Map copy(begin(rhs_arr), end(rhs_arr), std::less<int>(), lhs_alloc);
      copy = orig;
      assert(copy.size() == 3);
      assert(*std::next(copy.begin(), 0) == V(1, 1));
      assert(*std::next(copy.begin(), 1) == V(2, 3));
      assert(*std::next(copy.begin(), 2) == V(2, 6));
      assert(std::next(copy.begin(), 3) == copy.end());

      // Check that orig is still what is expected
      assert(orig.size() == 3);
      assert(*std::next(orig.begin(), 0) == V(1, 1));
      assert(*std::next(orig.begin(), 1) == V(2, 3));
      assert(*std::next(orig.begin(), 2) == V(2, 6));
      assert(std::next(orig.begin(), 3) == orig.end());
    }
    check_alloc_invariant();
  }
  { // Make a somewhat larger set to exercise the algorithm a bit
    using V   = std::pair<const int, int>;
    using Map = std::multimap<int, int, std::less<int>, Alloc>;

    Map orig(rhs_alloc);
    for (int i = 0; i != 50; ++i) {
      orig.insert(V(i, i + 3));
      orig.insert(V(i, i + 5));
    }

    Map copy(lhs_alloc);
    copy  = orig;
    int i = 0;
    for (auto iter = copy.begin(); iter != copy.end();) {
      assert(*iter++ == V(i, i + 3));
      assert(*iter++ == V(i, i + 5));
      ++i;
    }
  }
  check_alloc_invariant();
}

void test() {
  test_alloc<std::allocator<std::pair<const int, int> > >();
#if TEST_STD_VER >= 11
  test_alloc<min_allocator<std::pair<const int, int> > >();

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
    test_alloc<tracking_allocator<std::pair<const int, int> > >(
        lhs_allocs, rhs_allocs, AssertEmpty(lhs_allocs, rhs_allocs));
  }
#endif

  { // Ensure that the comparator is copied
    using V   = std::pair<const int, int>;
    using Map = std::multimap<int, int, test_less<int> >;

    V arr[] = {V(1, 1), V(2, 3), V(2, 6)};
    const Map orig(begin(arr), end(arr), test_less<int>(3));
    Map copy;
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
