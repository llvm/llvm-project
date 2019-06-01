//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class T, class U> bool operator<(const shared_ptr<T>& a, const shared_ptr<U>& b);

#include <memory>
#include <cassert>

#include "test_macros.h"

void do_nothing(int*) {}

int main(int, char**)
{
    int* ptr1(new int);
    int* ptr2(new int);
    const std::shared_ptr<int> p1(ptr1);
    const std::shared_ptr<int> p2(ptr2);
    const std::shared_ptr<int> p3(ptr2, do_nothing);
    assert((p1 < p2) == (ptr1 < ptr2));
    assert(!(p2 < p3) && !(p3 < p2));

  return 0;
}
