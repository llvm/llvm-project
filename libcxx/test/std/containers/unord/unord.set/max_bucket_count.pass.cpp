//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_set

// size_type max_bucket_count() const;

#include <unordered_set>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::unordered_set<int> C;
        const C c;
        assert(c.max_bucket_count() > 0);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_set<int, std::hash<int>,
                                      std::equal_to<int>, min_allocator<int>> C;
        const C c;
        assert(c.max_bucket_count() > 0);
    }
#endif

  return 0;
}
