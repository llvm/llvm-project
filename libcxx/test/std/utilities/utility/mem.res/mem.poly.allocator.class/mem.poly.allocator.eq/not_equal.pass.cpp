//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <memory_resource>

// template <class T> class polymorphic_allocator;

// template <class T>
// bool operator!=(
//      polymorphic_allocator<T> const &
//    , polymorphic_allocator<T> const &) noexcept

#include <memory_resource>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "test_std_memory_resource.h"

int main(int, char**) {
  typedef std::pmr::polymorphic_allocator<void> A1;
  typedef std::pmr::polymorphic_allocator<int> A2;
  // check return types
  {
    A1 const a1;
    A2 const a2;
    ASSERT_SAME_TYPE(decltype(a1 != a2), bool);
    ASSERT_NOEXCEPT(a1 != a2);
  }
  // not equal same type (different resource)
  {
    TestResource d1(1);
    TestResource d2(2);
    A1 const a1(&d1);
    A1 const a2(&d2);

    assert(a1 != a2);
    assert(d1.checkIsEqualCalledEq(1));
    assert(d2.checkIsEqualCalledEq(0));

    d1.reset();

    assert(a2 != a1);
    assert(d1.checkIsEqualCalledEq(0));
    assert(d2.checkIsEqualCalledEq(1));
  }
  // equal same type (same resource)
  {
    TestResource d1;
    A1 const a1(&d1);
    A1 const a2(&d1);

    assert(!(a1 != a2));
    assert(d1.checkIsEqualCalledEq(0));

    assert(!(a2 != a1));
    assert(d1.checkIsEqualCalledEq(0));
  }
  // equal same type
  {
    TestResource d1(1);
    TestResource d2(1);
    A1 const a1(&d1);
    A1 const a2(&d2);

    assert(!(a1 != a2));
    assert(d1.checkIsEqualCalledEq(1));
    assert(d2.checkIsEqualCalledEq(0));

    d1.reset();

    assert(!(a2 != a1));
    assert(d1.checkIsEqualCalledEq(0));
    assert(d2.checkIsEqualCalledEq(1));
  }
  // not equal different types
  {
    TestResource d1;
    TestResource1 d2;
    A1 const a1(&d1);
    A2 const a2(&d2);

    assert(a1 != a2);
    assert(d1.checkIsEqualCalledEq(1));
    assert(d2.checkIsEqualCalledEq(0));

    d1.reset();

    assert(a2 != a1);
    assert(d1.checkIsEqualCalledEq(0));
    assert(d2.checkIsEqualCalledEq(1));
  }

  return 0;
}
