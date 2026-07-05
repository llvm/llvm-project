//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T> class weak_ptr;
//
// not less than comparable

#include <memory>
#include <cassert>

int main(int, char**)
{
    const std::shared_ptr<int> p1(new int);
    const std::shared_ptr<int> p2(new int);
    const std::weak_ptr<int> w1(p1);
    const std::weak_ptr<int> w2(p2);

    bool b = w1 < w2;

  return 0;
}
