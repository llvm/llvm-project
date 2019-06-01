//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<const T> cref(const T& t);

#include <functional>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    int i = 0;
    std::reference_wrapper<const int> r = std::cref(i);
    assert(&r.get() == &i);

  return 0;
}
