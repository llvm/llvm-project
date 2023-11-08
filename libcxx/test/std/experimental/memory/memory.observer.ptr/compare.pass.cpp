// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// REQUIRES: c++experimental

// <experimental/memory>

// observer_ptr
//
// template <class W1, class W2>
// bool operator==(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2);
//
// template <class W1, class W2>
// bool operator!=(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2);
//
// template <class W>
// bool operator==(const observer_ptr<W>& p, std::nullptr_t) noexcept;
//
// template <class W>
// bool operator==(std::nullptr_t, const observer_ptr<W>& p) noexcept;
//
// template <class W>
// bool operator!=(const observer_ptr<W>& p, std::nullptr_t) noexcept;
//
// template <class W>
// bool operator!=(std::nullptr_t, const observer_ptr<W>& p) noexcept;
//
// template <class W1, class W2>
// bool operator<(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2);
//
// template <class W1, class W2>
// bool operator>(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2);
//
// template <class W1, class W2>
// bool operator<=(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2);
//
// template <class W1, class W2>
// bool operator>=(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2);

#include <experimental/memory>
#include <cassert>

void test() {
  using T       = int;
  using Ptr     = std::experimental::observer_ptr<T>;
  using VoidPtr = std::experimental::observer_ptr<void>;

  // operator==(observer_ptr, observer_ptr)
  {
    T obj1, obj2;
    Ptr ptr1(&obj1), ptr1_x(&obj1);
    Ptr ptr2(&obj2);
    VoidPtr ptr3(&obj1);

    assert(!(ptr1 == ptr2));
    assert(ptr1 == ptr1_x);

    assert(ptr1 == ptr3);
  }

  // operator!=(observer_ptr, observer_ptr)
  {
    T obj1, obj2;
    Ptr ptr1(&obj1), ptr1_x(&obj1);
    Ptr ptr2(&obj2);
    VoidPtr ptr3(&obj1);

    assert(ptr1 != ptr2);
    assert(!(ptr1 != ptr1_x));

    assert(ptr2 != ptr3);
  }

  // operator==(observer_ptr, nullptr_t)
  {
    T obj1;
    Ptr ptr1(&obj1);
    Ptr ptr2(nullptr);

    assert(!(ptr1 == nullptr));
    assert(ptr2 == nullptr);
  }

  // operator==(nullptr_t, observer_ptr)
  {
    T obj1;
    Ptr ptr1(&obj1);
    Ptr ptr2(nullptr);

    assert(!(nullptr == ptr1));
    assert(nullptr == ptr2);
  }

  // operator!=(observer_ptr, nullptr_t)
  {
    T obj1;
    Ptr ptr1(&obj1);
    Ptr ptr2(nullptr);

    assert(ptr1 != nullptr);
    assert(!(ptr2 != nullptr));
  }

  // operator!=(nullptr_t, observer_ptr)
  {
    T obj1;
    Ptr ptr1(&obj1);
    Ptr ptr2(nullptr);

    assert(nullptr != ptr1);
    assert(!(nullptr != ptr2));
  }

  // operator<(observer_ptr, observer_ptr)
  {
    T obj1, obj2;
    Ptr ptr1(&obj1);
    Ptr ptr2(&obj2);
    VoidPtr ptr3(&obj1);

    assert(!(ptr1 < ptr1));
    assert((ptr1 < ptr2) == (&obj1 < &obj2));

    assert(!(ptr1 < ptr3));
  }

  // operator>(observer_ptr, observer_ptr)
  {
    T obj1, obj2;
    Ptr ptr1(&obj1);
    Ptr ptr2(&obj2);
    VoidPtr ptr3(&obj1);

    assert(!(ptr1 > ptr1));
    assert((ptr1 > ptr2) == (&obj1 > &obj2));

    assert(!(ptr1 > ptr3));
  }

  // operator<=(observer_ptr, observer_ptr)
  {
    T obj1, obj2;
    Ptr ptr1(&obj1);
    Ptr ptr2(&obj2);
    VoidPtr ptr3(&obj1);

    assert(ptr1 <= ptr1);
    assert((ptr1 <= ptr2) == (&obj1 <= &obj2));

    assert(ptr1 <= ptr3);
  }

  // operator>=(observer_ptr, observer_ptr)
  {
    T obj1, obj2;
    Ptr ptr1(&obj1);
    Ptr ptr2(&obj2);
    VoidPtr ptr3(&obj1);

    assert(ptr1 >= ptr1);
    assert((ptr1 >= ptr2) == (&obj1 >= &obj2));

    assert(ptr1 >= ptr3);
  }
}

int main(int, char**) {
  // Note: this is not constexpr in the spec
  test();

  return 0;
}