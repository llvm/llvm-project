//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// std::find with vector<bool>::iterator

// https://bugs.llvm.org/show_bug.cgi?id=16816

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

int main(int, char**)
{
    {
        for (unsigned i = 1; i < 256; ++i)
        {
            std::vector<bool> b(i,true);
            std::vector<bool>::iterator j = std::find(b.begin()+1, b.end(), false);
            assert(static_cast<std::size_t>(j-b.begin()) == i);
            assert(b.end() == j);
        }
    }
    {
        for (unsigned i = 1; i < 256; ++i)
        {
            std::vector<bool> b(i,false);
            std::vector<bool>::iterator j = std::find(b.begin()+1, b.end(), true);
            assert(static_cast<std::size_t>(j-b.begin()) == i);
            assert(b.end() == j);
        }
    }

  return 0;
}
