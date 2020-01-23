//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <vector>

// vector& operator=(vector&& c);

#include <vector>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::vector<bool, test_allocator<bool> > l(test_allocator<bool>(5));
        std::vector<bool, test_allocator<bool> > lo(test_allocator<bool>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<bool, test_allocator<bool> > l2(test_allocator<bool>(5));
        l2 = std::move(l);
        assert(l2 == lo);
        LIBCPP_ASSERT(l.empty());
        assert(l2.get_allocator() == lo.get_allocator());
    }
    {
        std::vector<bool, test_allocator<bool> > l(test_allocator<bool>(5));
        std::vector<bool, test_allocator<bool> > lo(test_allocator<bool>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<bool, test_allocator<bool> > l2(test_allocator<bool>(6));
        l2 = std::move(l);
        assert(l2 == lo);
        assert(!l.empty());
        assert(l2.get_allocator() == test_allocator<bool>(6));
    }
    {
        std::vector<bool, other_allocator<bool> > l(other_allocator<bool>(5));
        std::vector<bool, other_allocator<bool> > lo(other_allocator<bool>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<bool, other_allocator<bool> > l2(other_allocator<bool>(6));
        l2 = std::move(l);
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == lo.get_allocator());
    }
    {
        std::vector<bool, min_allocator<bool> > l(min_allocator<bool>{});
        std::vector<bool, min_allocator<bool> > lo(min_allocator<bool>{});
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<bool, min_allocator<bool> > l2(min_allocator<bool>{});
        l2 = std::move(l);
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == lo.get_allocator());
    }

  return 0;
}
