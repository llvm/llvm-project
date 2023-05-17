//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test swap

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "deleter_types.h"

struct A
{
    int state_;
    static int count;
    TEST_CONSTEXPR_CXX23 A() : state_(0) {
      if (!TEST_IS_CONSTANT_EVALUATED)
        ++count;
    }
    TEST_CONSTEXPR_CXX23 explicit A(int i) : state_(i) {
      if (!TEST_IS_CONSTANT_EVALUATED)
        ++count;
    }
    TEST_CONSTEXPR_CXX23 A(const A& a) : state_(a.state_) {
      if (!TEST_IS_CONSTANT_EVALUATED)
        ++count;
    }
    TEST_CONSTEXPR_CXX23 A& operator=(const A& a) {
      state_ = a.state_;
      return *this;
    }
    TEST_CONSTEXPR_CXX23 ~A() {
      if (!TEST_IS_CONSTANT_EVALUATED)
        --count;
    }

    friend TEST_CONSTEXPR_CXX23 bool operator==(const A& x, const A& y) { return x.state_ == y.state_; }
};

int A::count = 0;

template <class T>
struct NonSwappableDeleter {
  TEST_CONSTEXPR_CXX23 explicit NonSwappableDeleter(int) {}
  TEST_CONSTEXPR_CXX23 NonSwappableDeleter& operator=(NonSwappableDeleter const&) { return *this; }
  TEST_CONSTEXPR_CXX23 void operator()(T*) const {}

private:
  NonSwappableDeleter(NonSwappableDeleter const&);

};

TEST_CONSTEXPR_CXX23 bool test() {
  {
    A* p1 = new A(1);
    std::unique_ptr<A, Deleter<A> > s1(p1, Deleter<A>(1));
    A* p2 = new A(2);
    std::unique_ptr<A, Deleter<A> > s2(p2, Deleter<A>(2));
    assert(s1.get() == p1);
    assert(*s1 == A(1));
    assert(s1.get_deleter().state() == 1);
    assert(s2.get() == p2);
    assert(*s2 == A(2));
    assert(s2.get_deleter().state() == 2);
    swap(s1, s2);
    assert(s1.get() == p2);
    assert(*s1 == A(2));
    assert(s1.get_deleter().state() == 2);
    assert(s2.get() == p1);
    assert(*s2 == A(1));
    assert(s2.get_deleter().state() == 1);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A::count == 2);
  }
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A::count == 0);
  {
    A* p1 = new A[3];
    std::unique_ptr<A[], Deleter<A[]> > s1(p1, Deleter<A[]>(1));
    A* p2 = new A[3];
    std::unique_ptr<A[], Deleter<A[]> > s2(p2, Deleter<A[]>(2));
    assert(s1.get() == p1);
    assert(s1.get_deleter().state() == 1);
    assert(s2.get() == p2);
    assert(s2.get_deleter().state() == 2);
    swap(s1, s2);
    assert(s1.get() == p2);
    assert(s1.get_deleter().state() == 2);
    assert(s2.get() == p1);
    assert(s2.get_deleter().state() == 1);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A::count == 6);
  }
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A::count == 0);
#if TEST_STD_VER >= 11
    {
        // test that unique_ptr's specialized swap is disabled when the deleter
        // is non-swappable. Instead we should pick up the generic swap(T, T)
        // and perform 3 move constructions.
        typedef NonSwappableDeleter<int> D;
        D  d(42);
        int x = 42;
        int y = 43;
        std::unique_ptr<int, D&> p(&x, d);
        std::unique_ptr<int, D&> p2(&y, d);
        std::swap(p, p2);
    }
#endif

    return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

  return 0;
}
