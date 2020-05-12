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

#include "test_macros.h"

#include <memory>
#include <type_traits>
#include <cassert>

struct A {
  int x;
};

int main(int, char**) {
  {
    const std::shared_ptr<A> pA(new A);
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
#if TEST_STD_VER > 14
  {
    const std::shared_ptr<A[8]> pA;
    std::shared_ptr<int[8]> pi = std::reinterpret_pointer_cast<int[8]>(pA);
    std::shared_ptr<A[8]> pA2 = std::reinterpret_pointer_cast<A[8]>(pi);
    assert(pA2.get() == pA.get());
    assert(!pi.owner_before(pA) && !pA.owner_before(pi));
  }
#endif // TEST_STD_VER > 14

  return 0;
}
