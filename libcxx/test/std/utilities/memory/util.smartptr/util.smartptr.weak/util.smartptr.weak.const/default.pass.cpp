//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class T> class weak_ptr

// weak_ptr();

#include <memory>
#include <cassert>

struct A;

int main()
{
    std::weak_ptr<A> p;
    assert(p.use_count() == 0);
}
