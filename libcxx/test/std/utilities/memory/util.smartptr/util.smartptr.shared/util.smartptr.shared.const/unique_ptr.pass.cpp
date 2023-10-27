//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: sanitizer-new-delete

// <memory>

// template <class Y, class D> shared_ptr(unique_ptr<Y, D>&&r);

#include <memory>
#include <new>
#include <cstdlib>
#include <cassert>
#include <utility>

#include "test_macros.h"
#include "count_new.h"

struct B
{
    static int count;

    B() {++count;}
    B(const B&) {++count;}
    virtual ~B() {--count;}
};

int B::count = 0;

struct A
    : public B
{
    static int count;

    A() {++count;}
    A(const A& other) : B(other) {++count;}
    ~A() {--count;}
};

int A::count = 0;

void fn ( const std::shared_ptr<int> &) {}
void fn ( const std::shared_ptr<B> &) { assert (false); }

template <typename T>
void assert_deleter ( T * ) { assert(false); }

namespace adl {
struct D {
    void operator()(int *) const {}
};
void ref(D);
}

template <class T>
struct StatefulDeleter {
  int state = 0;

  StatefulDeleter(int val = 0) : state(val) {}
  StatefulDeleter(StatefulDeleter const&) { assert(false); }

  void operator()(T* ptr) {
    assert(state == 42);
    delete ptr;
  }
};

template <class T>
struct StatefulArrayDeleter {
  int state = 0;

  StatefulArrayDeleter(int val = 0) : state(val) {}
  StatefulArrayDeleter(StatefulArrayDeleter const&) { assert(false); }

  void operator()(T* ptr) {
    assert(state == 42);
    delete []ptr;
  }
};

struct MovingDeleter {
  explicit MovingDeleter(int *moves) : moves_(moves) {}
  MovingDeleter(MovingDeleter&& rhs) : moves_(rhs.moves_) { *moves_ += 1; }
  void operator()(int*) const {}
  int *moves_;
};

// https://llvm.org/PR53368
// Bogus unique_ptr-to-shared_ptr conversions should be forbidden
#if TEST_STD_VER >= 17
static_assert( std::is_constructible<std::shared_ptr<A>,   std::unique_ptr<A>&&>::value, "");
static_assert( std::is_constructible<std::shared_ptr<A[]>, std::unique_ptr<A[]>&&>::value, "");
static_assert(!std::is_constructible<std::shared_ptr<A>,   std::unique_ptr<A[]>&&>::value, "");
static_assert(!std::is_constructible<std::shared_ptr<B[]>, std::unique_ptr<A[]>&&>::value, "");
static_assert(!std::is_constructible<std::shared_ptr<B>,   std::unique_ptr<A[]>&&>::value, "");
#endif

int main(int, char**)
{
    {
        std::unique_ptr<A> ptr(new A);
        A* raw_ptr = ptr.get();
        std::shared_ptr<B> p(std::move(ptr));
        assert(A::count == 1);
        assert(B::count == 1);
        assert(p.use_count() == 1);
        assert(p.get() == raw_ptr);
        assert(ptr.get() == 0);
    }

    {
        std::unique_ptr<A const> ptr(new A);
        A const* raw_ptr = ptr.get();
        std::shared_ptr<B const> p(std::move(ptr));
        assert(A::count == 1);
        assert(B::count == 1);
        assert(p.use_count() == 1);
        assert(p.get() == raw_ptr);
        assert(ptr.get() == 0);
    }

#ifndef TEST_HAS_NO_EXCEPTIONS
    assert(A::count == 0);
    {
        std::unique_ptr<A> ptr(new A);
        A* raw_ptr = ptr.get();
        globalMemCounter.throw_after = 0;
        try
        {
            std::shared_ptr<B> p(std::move(ptr));
            assert(false);
        }
        catch (...)
        {
            assert(A::count == 1);
            assert(B::count == 1);
            assert(ptr.get() == raw_ptr);
        }
    }
#endif

#if TEST_STD_VER > 14
    {
      std::unique_ptr<int> ptr;
      std::shared_ptr<int> p(std::move(ptr));
      assert(p.get() == 0);
      assert(p.use_count() == 0);
    }
#endif

    {
      StatefulDeleter<A> d;
      std::unique_ptr<A, StatefulDeleter<A>&> u(new A, d);
      std::shared_ptr<A> p(std::move(u));
      d.state = 42;
      assert(A::count == 1);
    }
    assert(A::count == 0);

    { // LWG 2399
        fn(std::unique_ptr<int>(new int));
    }
#if TEST_STD_VER >= 14
    { // LWG 2415
        std::unique_ptr<int, void (*)(int*)> p(nullptr, assert_deleter<int>);
        std::shared_ptr<int> p2(std::move(p)); // should not call deleter when going out of scope
    }
#endif

    {
    adl::D d;
    std::unique_ptr<int, adl::D&> u(nullptr, d);
    std::shared_ptr<int> s = std::move(u);
    }

    assert(A::count == 0);

#if TEST_STD_VER > 14
    {
      StatefulArrayDeleter<A> d;
      std::unique_ptr<A[], StatefulArrayDeleter<A>&> u(new A[4], d);
      std::shared_ptr<A[]> p(std::move(u));
      d.state = 42;
      assert(A::count == 4);
    }
    assert(A::count == 0);
    assert(B::count == 0);

    {
      std::unique_ptr<A[]> ptr(new A[8]);
      A* raw_ptr = ptr.get();
      std::shared_ptr<A[]> p(std::move(ptr));
      assert(A::count == 8);
      assert(p.use_count() == 1);
      assert(p.get() == raw_ptr);
      assert(ptr.get() == 0);
    }
    assert(A::count == 0);

    {
      int *p = new int[8];
      std::unique_ptr<int[]> u(p);
      std::shared_ptr<int[]> s(std::move(u));
      assert(u == nullptr);
      assert(s.get() == p);
    }
#endif // TEST_STD_VER > 14

    { // LWG 3548
      {
        int moves = 0;
        int i = 42;
        std::unique_ptr<int, MovingDeleter> u(&i, MovingDeleter(&moves));
        assert(moves == 1);
        std::shared_ptr<int> s(std::move(u));
        assert(moves >= 2);
        assert(u == nullptr);
        assert(s.get() == &i);
      }

#if TEST_STD_VER > 14
      {
        int moves = 0;
        int a[8];
        std::unique_ptr<int[], MovingDeleter> u(a, MovingDeleter(&moves));
        assert(moves == 1);
        std::shared_ptr<int[]> s = std::move(u);
        assert(moves >= 2);
        assert(u == nullptr);
        assert(s.get() == a);
      }
#endif // TEST_STD_VER > 14
    }

    return 0;
}
