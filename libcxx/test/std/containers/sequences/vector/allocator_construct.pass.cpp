//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <vector>

// Make sure that copying a range of trivially copyable elements into a vector calls the allocator's construct when it
// is usable with the source elements, even if it is a non-template member function.
// https://llvm.org/PR227115

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "test_macros.h"

struct Trivial {
  int value;
};

int construct_calls = 0;

template <class T>
struct ConstRefConstructAllocator {
  using value_type = T;

  ConstRefConstructAllocator() = default;
  template <class U>
  ConstRefConstructAllocator(const ConstRefConstructAllocator<U>&) {}

  T* allocate(std::size_t n) { return std::allocator<T>().allocate(n); }
  void deallocate(T* p, std::size_t n) { std::allocator<T>().deallocate(p, n); }

  void construct(T* p, const T& x) {
    ++construct_calls;
    ::new (static_cast<void*>(p)) T(x);
  }

  friend bool operator==(const ConstRefConstructAllocator&, const ConstRefConstructAllocator&) { return true; }
  friend bool operator!=(const ConstRefConstructAllocator&, const ConstRefConstructAllocator&) { return false; }
};

template <class T>
struct RefConstructAllocator {
  using value_type = T;

  RefConstructAllocator() = default;
  template <class U>
  RefConstructAllocator(const RefConstructAllocator<U>&) {}

  T* allocate(std::size_t n) { return std::allocator<T>().allocate(n); }
  void deallocate(T* p, std::size_t n) { std::allocator<T>().deallocate(p, n); }

  void construct(T* p, T& x) {
    ++construct_calls;
    ::new (static_cast<void*>(p)) T(x);
  }

  friend bool operator==(const RefConstructAllocator&, const RefConstructAllocator&) { return true; }
  friend bool operator!=(const RefConstructAllocator&, const RefConstructAllocator&) { return false; }
};

template <class Vec, class Elem>
void test_copy_from_range(Elem* first, Elem* last) {
  const int n = static_cast<int>(last - first);

  { // vector(first, last)
    construct_calls = 0;
    Vec v(first, last);
    assert(construct_calls == n);
    assert(static_cast<int>(v.size()) == n);
  }
  { // insert(pos, first, last) without reallocation
    Vec v;
    v.reserve(2 * n);
    construct_calls = 0;
    v.insert(v.end(), first, last);
    assert(construct_calls == n);
  }
  { // insert(pos, first, last) with reallocation
    Vec v;
    construct_calls = 0;
    v.insert(v.end(), first, last);
    assert(construct_calls == n);
  }
  { // assign(first, last)
    Vec v;
    construct_calls = 0;
    v.assign(first, last);
    assert(construct_calls == n);
  }
}

template <class T>
struct MoveConstructAllocator {
  using value_type = T;

  MoveConstructAllocator() = default;
  template <class U>
  MoveConstructAllocator(const MoveConstructAllocator<U>&) {}

  T* allocate(std::size_t n) { return std::allocator<T>().allocate(n); }
  void deallocate(T* p, std::size_t n) { std::allocator<T>().deallocate(p, n); }

  void construct(T* p, T&& x) { ::new (static_cast<void*>(p)) T(std::move(x)); }

  friend bool operator==(const MoveConstructAllocator&, const MoveConstructAllocator&) { return true; }
  friend bool operator!=(const MoveConstructAllocator&, const MoveConstructAllocator&) { return false; }
};

// Ranges of volatile elements must not be handed to memmove, which doesn't accept volatile pointers.
template <class Alloc>
void test_volatile_range() {
  using Vec = std::vector<int, Alloc>;

  volatile int arr[]  = {1, 2, 3};
  volatile int* first = arr;
  volatile int* last  = arr + 3;

  { // vector(first, last)
    Vec v(first, last);
    assert(v.size() == 3 && v[0] == 1 && v[1] == 2 && v[2] == 3);
  }
  { // insert(pos, first, last)
    Vec v;
    v.insert(v.end(), first, last);
    assert(v.size() == 3 && v[0] == 1 && v[1] == 2 && v[2] == 3);
  }
  { // assign(first, last) with and without reallocation
    Vec v;
    v.assign(first, last);
    assert(v.size() == 3 && v[0] == 1 && v[1] == 2 && v[2] == 3);
    v.assign(first, first + 2);
    assert(v.size() == 2 && v[0] == 1 && v[1] == 2);
  }
}

int main(int, char**) {
  test_volatile_range<std::allocator<int> >();
  test_volatile_range<MoveConstructAllocator<int> >();

  {
    using Vec = std::vector<Trivial, ConstRefConstructAllocator<Trivial> >;

    Trivial arr[]             = {{1}, {2}, {3}};
    const Trivial const_arr[] = {{1}, {2}, {3}};
    test_copy_from_range<Vec>(arr, arr + 3);
    test_copy_from_range<Vec>(const_arr, const_arr + 3);

    { // vector(initializer_list)
      construct_calls = 0;
      Vec v({{1}, {2}, {3}});
      assert(construct_calls == 3);
      assert(v.size() == 3);
    }
    { // insert(pos, initializer_list)
      Vec v;
      v.reserve(6);
      construct_calls = 0;
      v.insert(v.end(), {{1}, {2}, {3}});
      assert(construct_calls == 3);
    }
    { // copy constructor
      Vec v(arr, arr + 3);
      construct_calls = 0;
      Vec copy(v);
      assert(construct_calls == 3);
      assert(copy.size() == 3);
    }
  }
  {
    // construct(T*, T&) can only be used with non-const source elements.
    using Vec = std::vector<Trivial, RefConstructAllocator<Trivial> >;

    Trivial arr[] = {{1}, {2}, {3}};
    test_copy_from_range<Vec>(arr, arr + 3);
  }

  return 0;
}
