//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <vector>

// vector(vector&& c);

#include <vector>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX20 bool tests()
{
    test_allocator_statistics alloc_stats;
    {
        std::vector<bool, test_allocator<bool> > l(test_allocator<bool>(5, &alloc_stats));
        std::vector<bool, test_allocator<bool> > lo(test_allocator<bool>(5, &alloc_stats));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(true);
            lo.push_back(true);
        }
        std::vector<bool, test_allocator<bool> > l2 = std::move(l);
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == lo.get_allocator());
    }
    {
        std::vector<bool, other_allocator<bool> > l(other_allocator<bool>(5));
        std::vector<bool, other_allocator<bool> > lo(other_allocator<bool>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(true);
            lo.push_back(true);
        }
        std::vector<bool, other_allocator<bool> > l2 = std::move(l);
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == lo.get_allocator());
    }
    {
        std::vector<bool, min_allocator<bool> > l(min_allocator<bool>{});
        std::vector<bool, min_allocator<bool> > lo(min_allocator<bool>{});
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(true);
            lo.push_back(true);
        }
        std::vector<bool, min_allocator<bool> > l2 = std::move(l);
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == lo.get_allocator());
    }
    {
      alloc_stats.clear();
      using Vect = std::vector<bool, test_allocator<bool> >;
      using AllocT = Vect::allocator_type;
      Vect v(test_allocator<bool>(42, 101, &alloc_stats));
      assert(alloc_stats.count == 1);
      {
        const AllocT& a = v.get_allocator();
        assert(alloc_stats.count == 2);
        assert(a.get_data() == 42);
        assert(a.get_id() == 101);
      }
      assert(alloc_stats.count == 1);
      alloc_stats.clear_ctor_counters();

      Vect v2 = std::move(v);
      assert(alloc_stats.count == 2);
      assert(alloc_stats.copied == 0);
      assert(alloc_stats.moved == 1);
      {
        const AllocT& a1 = v.get_allocator();
        assert(a1.get_id() == test_alloc_base::moved_value);
        assert(a1.get_data() == 42);

        const AllocT& a2 = v2.get_allocator();
        assert(a2.get_id() == 101);
        assert(a2.get_data() == 42);

        assert(a1 == a2);
      }
    }

    return true;
}

int main(int, char**)
{
    tests();
#if TEST_STD_VER > 17
    static_assert(tests());
#endif
    return 0;
}
