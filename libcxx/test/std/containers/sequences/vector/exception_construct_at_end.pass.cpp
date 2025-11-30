//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// UNSUPPORTED: c++03

// <vector>

// Make sure elements are destroyed when exceptions thrown in __construct_at_end

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

#include "test_macros.h"
#if TEST_STD_VER >= 20
#  include <ranges>
#endif

#include "common.h"
#include "count_new.h"

#ifdef DISABLE_NEW_COUNT
#  define CHECK_NEW_DELETE_DIFF(...)
#else
#  define CHECK_NEW_DELETE_DIFF(__n) assert(globalMemCounter.new_called == globalMemCounter.delete_called + __n)
#endif

struct throw_context {
  static int num;
  static int limit;

  throw_context(int lim = 2) {
    num   = 0;
    limit = lim;
  }

  static void inc() {
    ++num;
    if (num >= limit) {
      --num;
      throw 1;
    }
  }

  static void dec() { --num; }
};

int throw_context::num   = 0;
int throw_context::limit = 0;

int debug = 0;

class throw_element {
public:
  throw_element() : data(new int(1)) {
    try {
      throw_context::inc();
    } catch (int) {
      delete data;
      throw;
    }
  }

  throw_element(throw_element const&) : data(new int(1)) {
    try {
      throw_context::inc();
    } catch (int) {
      delete data;
      throw;
    }
  }

  ~throw_element() {
    if (data) {
      delete data;
      throw_context::dec();
      if (debug)
        printf("dctor\n");
    }
  }

  throw_element& operator=(throw_element const&) { return *this; }

private:
  int* data;
};

int main(int, char*[]) {
  using AllocType = std::allocator<throw_element>;

  // vector(size_type __n)
  {
    throw_context ctx;
    try {
      std::vector<throw_element> v(3);
    } catch (int) {
    }
    check_new_delete_called();
  }

  // TODO: This constructor may be applied to C++11 according to https://cplusplus.github.io/LWG/issue2210
#if TEST_STD_VER >= 14
  // vector(size_type __n, const allocator_type& __a)
  {
    throw_context ctx;
    AllocType alloc;
    try {
      std::vector<throw_element> v(3, alloc);
    } catch (int) {
    }
    check_new_delete_called();
  }
#endif

  // vector(size_type __n, const value_type& __x)
  {
    throw_context ctx(3);
    try {
      throw_element e;
      std::vector<throw_element> v(3, e);
    } catch (int) {
    }
    check_new_delete_called();
  }

  // vector(size_type __n, const value_type& __x, const allocator_type& __a)
  {
    throw_context ctx(3);
    try {
      throw_element e;
      AllocType alloc;
      std::vector<throw_element> v(4, e, alloc);
    } catch (int) {
    }
    check_new_delete_called();
  }

  // vector(_ForwardIterator __first, _ForwardIterator __last)
  {
    throw_context ctx(4);
    try {
      std::vector<throw_element> v1(2);
      std::vector<throw_element> v2(v1.begin(), v1.end());
    } catch (int) {
    }
    check_new_delete_called();
  }

  // vector(_ForwardIterator __first, _ForwardIterator __last, const allocator_type& __a)
  {
    throw_context ctx(4);
    AllocType alloc;
    try {
      std::vector<throw_element> v1(2);
      std::vector<throw_element> v2(v1.begin(), v1.end(), alloc);
    } catch (int) {
    }
    check_new_delete_called();
  }

#if TEST_STD_VER >= 23
  // vector(from_range_t, _Range&& __range, const allocator_type& __alloc = allocator_type())
  {
    throw_context ctx(4);
    try {
      std::vector<throw_element> r(2);
      std::vector<throw_element> v(std::from_range, std::views::counted(r.begin(), 2));
    } catch (int) {
    }
    check_new_delete_called();
  }
#endif

  // vector(const vector& __x)
  {
    throw_context ctx(4);
    try {
      std::vector<throw_element> v1(2);
      std::vector<throw_element> v2(v1);
    } catch (int) {
    }
    check_new_delete_called();
  }

#if TEST_STD_VER >= 11
  // vector(initializer_list<value_type> __il)
  {
    throw_context ctx(6);
    try {
      throw_element e;
      std::vector<throw_element> v({e, e, e});
    } catch (int) {
    }
    check_new_delete_called();
  }

  // vector(initializer_list<value_type> __il, const allocator_type& __a)
  {
    throw_context ctx(6);
    AllocType alloc;
    try {
      throw_element e;
      std::vector<throw_element> v({e, e, e}, alloc);
    } catch (int) {
    }
    check_new_delete_called();
  }
#endif

  // void resize(size_type __sz)
  {
    // cap < size
    throw_context ctx;
    std::vector<throw_element> v;
    v.reserve(5);
    try {
      v.resize(4);
    } catch (int) {
    }
    CHECK_NEW_DELETE_DIFF(1);
  }
  check_new_delete_called();

  // void resize(size_type __sz, const_reference __x)
  {
    // cap < size
    throw_context ctx(3);
    std::vector<throw_element> v;
    v.reserve(5);
    try {
      throw_element e;
      v.resize(4, e);
    } catch (int) {
    }
    CHECK_NEW_DELETE_DIFF(1);
  }
  check_new_delete_called();

  // void assign(_ForwardIterator __first, _ForwardIterator __last)
  {
    // new size <= cap && new size > size
    throw_context ctx(4);
    std::vector<throw_element> v;
    v.reserve(3);
    try {
      std::vector<throw_element> data(2);
      v.assign(data.begin(), data.end());
    } catch (int) {
    }
    CHECK_NEW_DELETE_DIFF(1);
  }
  check_new_delete_called();

  {
    // new size > cap
    throw_context ctx(4);
    std::vector<throw_element> v;
    try {
      std::vector<throw_element> data(2);
      v.assign(data.begin(), data.end());
    } catch (int) {
    }
    CHECK_NEW_DELETE_DIFF(1);
  }
  check_new_delete_called();

#if TEST_STD_VER >= 23
  // void assign_range(_Range&& __range)
  {
    throw_context ctx(5);
    std::vector<throw_element> v;
    try {
      std::vector<throw_element> r(3);
      v.assign_range(r);
    } catch (int) {
    }
    CHECK_NEW_DELETE_DIFF(1);
  }
  check_new_delete_called();
#endif

#if TEST_STD_VER >= 11
  // vector& operator=(initializer_list<value_type> __il)
  {
    throw_context ctx(5);
    std::vector<throw_element> v;
    try {
      throw_element e;
      v = {e, e};
    } catch (int) {
    }
    CHECK_NEW_DELETE_DIFF(1);
  }
  check_new_delete_called();
#endif

  // vector<_Tp, _Allocator>& vector<_Tp, _Allocator>::operator=(const vector& __x)
  {
    throw_context ctx(4);
    std::vector<throw_element> v;
    try {
      std::vector<throw_element> data(2);
      v = data;
    } catch (int) {
    }
    CHECK_NEW_DELETE_DIFF(1);
  }
  check_new_delete_called();

  // iterator insert(const_iterator __position, _ForwardIterator __first, _ForwardIterator __last)
  {
    throw_context ctx(6);
    std::vector<throw_element> v;
    v.reserve(10);
    try {
      std::vector<throw_element> data(3);
      v.insert(v.begin(), data.begin(), data.end());
    } catch (int) {
    }
    CHECK_NEW_DELETE_DIFF(1);
  }
  check_new_delete_called();

#if TEST_STD_VER >= 23
  // iterator insert_range(const_iterator __position, _Range&& __range)
  {
    throw_context ctx(3);
    std::vector<throw_element> v;
    try {
      std::vector<throw_element> data(2);
      v.insert_range(v.begin(), data);
    } catch (int) {
    }
    check_new_delete_called();
  }
#endif

  return 0;
}
