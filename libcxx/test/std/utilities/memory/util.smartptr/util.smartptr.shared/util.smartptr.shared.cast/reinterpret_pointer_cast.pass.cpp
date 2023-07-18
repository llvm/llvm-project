//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class T, class U>
//     shared_ptr<T> reinterpret_pointer_cast(const shared_ptr<U>& r) noexcept;
// template<class T, class U>
//     shared_ptr<T> reinterpret_pointer_cast(shared_ptr<U>&& r) noexcept;

#include "test_macros.h"

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

struct A {
  int x;
};

struct Base { };
struct Derived : public Base { };

int main(int, char**) {
  {
    const std::shared_ptr<A> pA(new A);
    ASSERT_NOEXCEPT(std::reinterpret_pointer_cast<int>(pA));
    std::shared_ptr<int> pi = std::reinterpret_pointer_cast<int>(pA);
    std::shared_ptr<A> pA2 = std::reinterpret_pointer_cast<A>(pi);
    assert(pA2.get() == pA.get());
    assert(!pi.owner_before(pA) && !pA.owner_before(pi));
  }
  {
    const std::shared_ptr<A> pA;
    std::shared_ptr<int> pi = std::reinterpret_pointer_cast<int>(pA);
    std::shared_ptr<A> pA2 = std::reinterpret_pointer_cast<A>(pi);
    assert(pA2.get() == pA.get());
    assert(!pi.owner_before(pA) && !pA.owner_before(pi));
  }
  {
    const std::shared_ptr<A> pA(new A);
    std::shared_ptr<int> pi = std::reinterpret_pointer_cast<int>(pA);
    pA->x = 42;
    assert(*pi == 42);
  }
  {
    const std::shared_ptr<Derived> pDerived(new Derived);
    std::shared_ptr<Base> pBase = std::reinterpret_pointer_cast<Base>(pDerived);
    std::shared_ptr<Derived> pDerived2 = std::reinterpret_pointer_cast<Derived>(pBase);
    assert(pDerived2.get() == pDerived2.get());
    assert(!pBase.owner_before(pDerived) && !pDerived.owner_before(pBase));
  }
  {
    const std::shared_ptr<Base> pBase(new Base);
    std::shared_ptr<Derived> pDerived = std::reinterpret_pointer_cast<Derived>(pBase);
    std::shared_ptr<Base> pBase2 = std::reinterpret_pointer_cast<Base>(pDerived);
    assert(pBase2.get() == pBase.get());
    assert(!pDerived.owner_before(pBase) && !pBase.owner_before(pDerived));
  }
#if TEST_STD_VER > 14
  {
    const std::shared_ptr<A[8]> pA;
    std::shared_ptr<int[8]> pi = std::reinterpret_pointer_cast<int[8]>(pA);
    std::shared_ptr<A[8]> pA2 = std::reinterpret_pointer_cast<A[8]>(pi);
    assert(pA2.get() == pA.get());
    assert(!pi.owner_before(pA) && !pA.owner_before(pi));
  }
#endif // TEST_STD_VER > 14
#if TEST_STD_VER > 20
  {
    A* pA_raw = new A;
    std::shared_ptr<A> pA(pA_raw);
    ASSERT_NOEXCEPT(std::reinterpret_pointer_cast<int>(std::move(pA)));
    std::shared_ptr<int> pi = std::reinterpret_pointer_cast<int>(std::move(pA));
    assert(pA.get() == nullptr);
    assert(pi.use_count() == 1);
    std::shared_ptr<A> pA2 = std::reinterpret_pointer_cast<A>(std::move(pi));
    assert(pi.get() == nullptr);
    assert(pA2.get() == pA_raw);
    assert(pA2.use_count() == 1);
  }
#endif // TEST_STD_VER > 20

  return 0;
}
