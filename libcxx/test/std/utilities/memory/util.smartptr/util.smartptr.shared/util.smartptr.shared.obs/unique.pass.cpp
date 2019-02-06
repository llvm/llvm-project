//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// bool unique() const;

#include <memory>
#include <cassert>

int main()
{
    const std::shared_ptr<int> p(new int(32));
    assert(p.unique());
    {
    std::shared_ptr<int> p2 = p;
    assert(!p.unique());
    }
    assert(p.unique());
}
