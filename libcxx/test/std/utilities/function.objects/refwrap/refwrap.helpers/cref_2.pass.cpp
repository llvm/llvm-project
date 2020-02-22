//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<const T> cref(reference_wrapper<T> t);

#include <functional>
#include <cassert>

#include "test_macros.h"

namespace adl {
  struct A {};
  void cref(A) {}
}

int main(int, char**)
{
    const int i = 0;
    std::reference_wrapper<const int> r1 = std::cref(i);
    std::reference_wrapper<const int> r2 = std::cref(r1);
    assert(&r2.get() == &i);

    {
    adl::A a;
    std::reference_wrapper<const adl::A> a1 = std::cref(a);
    std::reference_wrapper<const adl::A> a2 = std::cref(a1);
    assert(&a2.get() == &a);
    }

  return 0;
}
