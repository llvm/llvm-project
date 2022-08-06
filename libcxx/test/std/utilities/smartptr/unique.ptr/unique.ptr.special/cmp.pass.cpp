//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator==(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator!=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator< (const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator> (const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator<=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator>=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template<class T1, class D1, class T2, class D2>
//   requires three_way_comparable_with<typename unique_ptr<T1, D1>::pointer,
//                                      typename unique_ptr<T2, D2>::pointer>
//   compare_three_way_result_t<typename unique_ptr<T1, D1>::pointer,
//                              typename unique_ptr<T2, D2>::pointer>
//     operator<=>(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "deleter_types.h"
#include "test_comparisons.h"

struct A {
  static int count;
  A() { ++count; }
  A(const A&) { ++count; }
  virtual ~A() { --count; }
};

int A::count = 0;

struct B : public A {
  static int count;
  B() { ++count; }
  B(const B& other) : A(other) { ++count; }
  virtual ~B() { --count; }
};

int B::count = 0;

int main(int, char**) {
  AssertComparisonsReturnBool<std::unique_ptr<int> >();
#if TEST_STD_VER > 17
  AssertOrderReturn<std::strong_ordering, std::unique_ptr<int>>();
#endif

  // Pointers of same type
  {
    A* ptr1 = new A;
    A* ptr2 = new A;
    const std::unique_ptr<A, Deleter<A> > p1(ptr1);
    const std::unique_ptr<A, Deleter<A> > p2(ptr2);

    assert(!(p1 == p2));
    assert(p1 != p2);
    assert((p1 < p2) == (ptr1 < ptr2));
    assert((p1 <= p2) == (ptr1 <= ptr2));
    assert((p1 > p2) == (ptr1 > ptr2));
    assert((p1 >= p2) == (ptr1 >= ptr2));
#if TEST_STD_VER > 17
    assert((p1 <=> p2) != std::strong_ordering::equal);
    assert((p1 <=> p2) == (ptr1 <=> ptr2));
#endif
  }
  // Pointers of different type
  {
    A* ptr1 = new A;
    B* ptr2 = new B;
    const std::unique_ptr<A, Deleter<A> > p1(ptr1);
    const std::unique_ptr<B, Deleter<B> > p2(ptr2);
    assert(!(p1 == p2));
    assert(p1 != p2);
    assert((p1 < p2) == (ptr1 < ptr2));
    assert((p1 <= p2) == (ptr1 <= ptr2));
    assert((p1 > p2) == (ptr1 > ptr2));
    assert((p1 >= p2) == (ptr1 >= ptr2));
#if TEST_STD_VER > 17
    assert((p1 <=> p2) != std::strong_ordering::equal);
    assert((p1 <=> p2) == (ptr1 <=> ptr2));
#endif
  }
  // Pointers of same array type
  {
    A* ptr1 = new A[3];
    A* ptr2 = new A[3];
    const std::unique_ptr<A[], Deleter<A[]> > p1(ptr1);
    const std::unique_ptr<A[], Deleter<A[]> > p2(ptr2);
    assert(!(p1 == p2));
    assert(p1 != p2);
    assert((p1 < p2) == (ptr1 < ptr2));
    assert((p1 <= p2) == (ptr1 <= ptr2));
    assert((p1 > p2) == (ptr1 > ptr2));
    assert((p1 >= p2) == (ptr1 >= ptr2));
#if TEST_STD_VER > 17
    assert((p1 <=> p2) != std::strong_ordering::equal);
    assert((p1 <=> p2) == (ptr1 <=> ptr2));
#endif
  }
  // Pointers of different array types
  {
    A* ptr1 = new A[3];
    B* ptr2 = new B[3];
    const std::unique_ptr<A[], Deleter<A[]> > p1(ptr1);
    const std::unique_ptr<B[], Deleter<B[]> > p2(ptr2);
    assert(!(p1 == p2));
    assert(p1 != p2);
    assert((p1 < p2) == (ptr1 < ptr2));
    assert((p1 <= p2) == (ptr1 <= ptr2));
    assert((p1 > p2) == (ptr1 > ptr2));
    assert((p1 >= p2) == (ptr1 >= ptr2));
#if TEST_STD_VER > 17
    assert((p1 <=> p2) != std::strong_ordering::equal);
    assert((p1 <=> p2) == (ptr1 <=> ptr2));
#endif
  }
  // Default-constructed pointers of same type
  {
    const std::unique_ptr<A, Deleter<A> > p1;
    const std::unique_ptr<A, Deleter<A> > p2;
    assert(p1 == p2);
#if TEST_STD_VER > 17
    assert((p1 <=> p2) == std::strong_ordering::equal);
#endif
  }
  // Default-constructed pointers of different type
  {
    const std::unique_ptr<A, Deleter<A> > p1;
    const std::unique_ptr<B, Deleter<B> > p2;
    assert(p1 == p2);
#if TEST_STD_VER > 17
    assert((p1 <=> p2) == std::strong_ordering::equal);
#endif
  }

  return 0;
}
