//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class T, class A, class... Args>
// shared_ptr<T> allocate_shared(const A& a, Args&&... args);

// This test checks that allocator_traits::construct and allocator_traits::destroy
// are used in allocate_shared as requested for the resolution of LWG2070. Note
// that LWG2070 was resolved by P0674R1 (which is a C++20 paper), but we implement
// LWG issue resolutions as DRs per our policy.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#include "test_macros.h"

static bool construct_called = false;
static bool destroy_called = false;
static unsigned allocator_id = 0;

template <class T>
struct MyAllocator {
public:
  typedef T value_type;
  typedef T* pointer;

  unsigned id = 0;

  MyAllocator() = default;
  MyAllocator(int i) : id(i) {}

  template <class U>
  MyAllocator(MyAllocator<U> const& other) : id(other.id){};

  pointer allocate(std::ptrdiff_t n) {
    return pointer(static_cast<T*>(::operator new(n * sizeof(T))));
  }

  void deallocate(pointer p, std::ptrdiff_t) { return ::operator delete(p); }

  template <typename ...Args>
  void construct(T* p, Args&& ...args) {
    construct_called = true;
    destroy_called = false;
    allocator_id = id;
    ::new (p) T(std::forward<Args>(args)...);
  }

  void destroy(T* p) {
    construct_called = false;
    destroy_called = true;
    allocator_id = id;
    p->~T();
  }
};

struct Private;

class Factory {
public:
  static std::shared_ptr<Private> allocate();
};

template <class T>
struct FactoryAllocator;

struct Private {
  int id;

private:
  friend FactoryAllocator<Private>;
  Private(int i) : id(i) {}
  ~Private() {}
};

template <class T>
struct FactoryAllocator : std::allocator<T> {
  FactoryAllocator() = default;

  template <class T1>
  FactoryAllocator(FactoryAllocator<T1>) {}

  template <class T1>
  struct rebind {
    typedef FactoryAllocator<T1> other;
  };

  void construct(void* p, int id) { ::new (p) Private(id); }
  void destroy(Private* p) { p->~Private(); }
};

std::shared_ptr<Private> Factory::allocate() {
  FactoryAllocator<Private> factory_alloc;
  return std::allocate_shared<Private>(factory_alloc, 42);
}

struct mchar {
  char c;
};

struct Foo {
  int val;

  Foo(int v) : val(v) {}

  Foo(Foo a, Foo b) : val(a.val + b.val) {}
};

void test_aligned(void* p, std::size_t align) {
  assert(reinterpret_cast<std::uintptr_t>(p) % align == 0);
}

int main(int, char**) {
  {
    std::shared_ptr<int> p = std::allocate_shared<int>(MyAllocator<int>());
    assert(construct_called);
  }
  assert(destroy_called);
  {
    std::shared_ptr<Foo> p =
        std::allocate_shared<Foo>(MyAllocator<Foo>(), Foo(42), Foo(100));
    assert(construct_called);
    assert(p->val == 142);
  }
  assert(destroy_called);
  { // Make sure allocator is copied.
    std::shared_ptr<int> p = std::allocate_shared<int>(MyAllocator<int>(3));
    assert(allocator_id == 3);

    allocator_id = 0;
  }
  assert(allocator_id == 3);

  {
    std::shared_ptr<int> p = std::allocate_shared<int>(MyAllocator<int>(), 42);
    assert(construct_called);
    assert(*p == 42);
  }
  assert(destroy_called);

  { // Make sure allocator is properly re-bound.
    std::shared_ptr<int> p =
        std::allocate_shared<int>(MyAllocator<mchar>(), 42);
    assert(construct_called);
    assert(*p == 42);
  }
  assert(destroy_called);

  {
    // Make sure that we call the correct allocator::construct. Private has a private constructor
    // so the construct method must be called on its friend Factory's allocator
    // (Factory::Allocator).
    std::shared_ptr<Private> p = Factory().allocate();
    assert(p->id == 42);
  }

#if TEST_STD_VER >= 11
  {
    struct Bar {
      std::max_align_t y;
    };

    std::shared_ptr<Bar> p;
    test_aligned(p.get(), alignof(Bar));
  }
#endif

  return 0;
}
