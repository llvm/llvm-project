//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<T> ref(T& t);

#include <functional>
#include <cassert>

int main(int, char**)
{
    int i = 0;
    std::reference_wrapper<int> r = std::ref(i);
    assert(&r.get() == &i);

  return 0;
}
