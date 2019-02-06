//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// T& operator*() const;

#include <memory>
#include <cassert>

int main()
{
    const std::shared_ptr<int> p(new int(32));
    assert(*p == 32);
    *p = 3;
    assert(*p == 3);
}
