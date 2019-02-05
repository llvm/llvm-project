//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// explicit operator bool() const;

#include <memory>
#include <cassert>

int main(int, char**)
{
    {
    const std::shared_ptr<int> p(new int(32));
    assert(p);
    }
    {
    const std::shared_ptr<int> p;
    assert(!p);
    }

  return 0;
}
