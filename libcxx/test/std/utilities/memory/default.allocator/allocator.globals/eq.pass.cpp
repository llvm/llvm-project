//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:

// template <class T1, class T2>
//   bool
//   operator==(const allocator<T1>&, const allocator<T2>&) throw();
//
// template <class T1, class T2>
//   bool
//   operator!=(const allocator<T1>&, const allocator<T2>&) throw();

#include <memory>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::allocator<int> a1;
    std::allocator<int> a2;
    assert(a1 == a2);
    assert(!(a1 != a2));

  return 0;
}
