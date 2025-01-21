//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// <vector>

// Check that vector assignments do not alter the RHS vector when an exception is thrown during reallocations triggered by assignments.

#include <cassert>
#include <ranges>
#include <vector>

#include "allocators.h"
#include "exception_test_helpers.h"
#include "test_allocator.h"
#include "test_macros.h"

void test_allocation_exception() {
#if TEST_STD_VER >= 14
  {
    limited_alloc_wrapper<int> alloc1 = limited_allocator<int, 100>();
    limited_alloc_wrapper<int> alloc2 = limited_allocator<int, 200>();
    std::vector<int, limited_alloc_wrapper<int> > v(100, alloc1);
    std::vector<int, limited_alloc_wrapper<int> > in(200, alloc2);
    try { // Throw in copy-assignment operator during allocation
      v = in;
    } catch (const std::exception&) {
    }
    assert(v.size() == 100);
  }

  {
    limited_alloc_wrapper<int> alloc1 = limited_allocator<int, 100>();
    limited_alloc_wrapper<int> alloc2 = limited_allocator<int, 200>();
    std::vector<int, limited_alloc_wrapper<int> > v(100, alloc1);
    std::vector<int, limited_alloc_wrapper<int> > in(200, alloc2);
    try { // Throw in move-assignment operator during allocation
      v = std::move(in);
    } catch (const std::exception&) {
    }
    assert(v.size() == 100);
  }
#endif

#if TEST_STD_VER >= 11
  {
    std::vector<int, limited_allocator<int, 5> > v(5);
    std::initializer_list<int> in{1, 2, 3, 4, 5, 6};
    try { // Throw in operator=(initializer_list<value_type>) during allocation
      v = in;
    } catch (const std::exception&) {
    }
    assert(v.size() == 5);
  }

  {
    std::vector<int, limited_allocator<int, 5> > v(5);
    std::initializer_list<int> in{1, 2, 3, 4, 5, 6};
    try { // Throw in assign(initializer_list<value_type>) during allocation
      v.assign(in);
    } catch (const std::exception&) {
    }
    assert(v.size() == 5);
  }
#endif

  {
    std::vector<int, limited_allocator<int, 100> > v(100);
    std::vector<int> in(101, 1);
    try { // Throw in assign(_ForwardIterator, _ForwardIterator) during allocation
      v.assign(in.begin(), in.end());
    } catch (const std::exception&) {
    }
    assert(v.size() == 100);
  }

  {
    std::vector<int, limited_allocator<int, 100> > v(100);
    try { // Throw in assign(size_type, const_reference) during allocation
      v.assign(101, 1);
    } catch (const std::exception&) {
    }
    assert(v.size() == 100);
  }

#if TEST_STD_VER >= 23
  {
    std::vector<int, limited_allocator<int, 100> > v(100);
    std::vector<int> in(101, 1);
    try { // Throw in assign(_ForwardIterator, _ForwardIterator) during allocation
      v.assign_range(in);
    } catch (const std::exception&) {
    }
    assert(v.size() == 100);
  }
#endif
}

void test_construction_exception() {
  {
    int throw_after = 10;
    throwing_t t    = throw_after;
    std::vector<throwing_t> in(6, t);
    std::vector<throwing_t> v(3, t);
    try { // Throw in copy-assignment operator from element type during construction
      v = in;
    } catch (int) {
    }
    assert(v.size() == 3);
  }

#if TEST_STD_VER >= 11
  {
    int throw_after = 10;
    throwing_t t    = throw_after;
    NONPOCMAAllocator<throwing_t> alloc1(1);
    NONPOCMAAllocator<throwing_t> alloc2(2);
    std::vector<throwing_t, NONPOCMAAllocator<throwing_t> > in(6, t, alloc1);
    std::vector<throwing_t, NONPOCMAAllocator<throwing_t> > v(3, t, alloc2);
    try { // Throw in move-assignment operator from element type during construction
      v = std::move(in);
    } catch (int) {
    }
    assert(v.size() == 3);
  }

  {
    int throw_after = 10;
    throwing_t t    = throw_after;
    std::initializer_list<throwing_t> in{t, t, t, t, t, t};
    std::vector<throwing_t> v(3, t);
    try { // Throw in operator=(initializer_list<value_type>) from element type during construction
      v = in;
    } catch (int) {
    }
    assert(v.size() == 3);
  }

  {
    int throw_after = 10;
    throwing_t t    = throw_after;
    std::initializer_list<throwing_t> in{t, t, t, t, t, t};
    std::vector<throwing_t> v(3, t);
    try { // Throw in assign(initializer_list<value_type>) from element type during construction
      v.assign(in);
    } catch (int) {
    }
    assert(v.size() == 3);
  }
#endif

  {
    std::vector<int> v(3);
    try { // Throw in assign(_ForwardIterator, _ForwardIterator) from forward iterator during construction
      v.assign(throwing_iterator<int, std::forward_iterator_tag>(),
               throwing_iterator<int, std::forward_iterator_tag>(6));
    } catch (int) {
    }
    assert(v.size() == 3);
  }

  {
    int throw_after = 10;
    throwing_t t    = throw_after;
    std::vector<throwing_t> in(6, t);
    std::vector<throwing_t> v(3, t);
    try { // Throw in assign(_ForwardIterator, _ForwardIterator) from element type during construction
      v.assign(in.begin(), in.end());
    } catch (int) {
    }
    assert(v.size() == 3);
  }

#if TEST_STD_VER >= 23
  {
    int throw_after = 10;
    throwing_t t    = throw_after;
    std::vector<throwing_t> in(6, t);
    std::vector<throwing_t> v(3, t);
    try { // Throw in assign_range(_Range&&) from element type during construction
      v.assign_range(in);
    } catch (int) {
    }
    assert(v.size() == 3);
  }
#endif

  {
    int throw_after = 4;
    throwing_t t    = throw_after;
    std::vector<throwing_t> v(3, t);
    try { // Throw in assign(size_type, const_reference) from element type during construction
      v.assign(6, t);
    } catch (int) {
    }
    assert(v.size() == 3);
  }
}

int main() {
  test_allocation_exception();
  test_construction_exception();
}
