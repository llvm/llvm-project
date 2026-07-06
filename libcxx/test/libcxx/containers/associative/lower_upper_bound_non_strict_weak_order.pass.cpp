//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>
// <map>

// This test ensures that libc++ maintains its historical behavior when std::set
// or std::map are used with a comparator that isn't quite a strict weak ordering
// when _LIBCPP_ENABLE_LEGACY_TREE_LOWER_UPPER_BOUND is defined.
//
// This escape hatch is only supported as a temporary means to give a bit more time
// for codebases to fix incorrect usages of associative containers.

// This precise test reproduces a comparator similar to boost::icl::exclusive_less
// for discrete right-open intervals: it orders disjoint intervals but treats
// overlapping intervals as equivalent, which is incorrect but was relied upon.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_LEGACY_TREE_LOWER_UPPER_BOUND
// UNSUPPORTED: c++03, c++11, c++14

#include <cassert>
#include <map>
#include <set>
#include <utility>

struct Interval {
  int lower;
  int upper;
};

struct ExclusiveLess {
  bool operator()(const Interval& lhs, const Interval& rhs) const { return lhs.upper <= rhs.lower; }
};

Interval right_open(int lo, int hi) { return Interval{lo, hi}; }

int main() {
  // std::set
  {
    // upper_bound
    {
      std::set<Interval, ExclusiveLess> intervals = {
          right_open(0, 2),
          right_open(3, 5),
          right_open(6, 9),
          right_open(10, 12),
      };
      // non-const
      {
        auto it = intervals.upper_bound(right_open(4, 7));
        assert(it != intervals.end());
        assert(it->lower == 10);
        assert(it->upper == 12);
      }

      // const
      {
        auto it = std::as_const(intervals).upper_bound(right_open(4, 7));
        assert(it != intervals.end());
        assert(it->lower == 10);
        assert(it->upper == 12);
      }
    }

    // lower_bound
    {
      std::set<Interval, ExclusiveLess> intervals = {
          right_open(0, 2),
          right_open(3, 5),
          right_open(6, 9),
          right_open(10, 12),
      };
      // non-const
      {
        auto it = intervals.lower_bound(right_open(1, 4));
        assert(it != intervals.end());
        assert(it->lower == 0);
        assert(it->upper == 2);
      }

      // const
      {
        auto it = std::as_const(intervals).lower_bound(right_open(1, 4));
        assert(it != intervals.end());
        assert(it->lower == 0);
        assert(it->upper == 2);
      }
    }
  }

  // std::map
  {
    using X   = int;
    X const x = 99;

    // upper_bound
    {
      std::map<Interval, X, ExclusiveLess> intervals = {
          {right_open(0, 2), x},
          {right_open(3, 5), x},
          {right_open(6, 9), x},
          {right_open(10, 12), x},
      };
      // non-const
      {
        auto it = intervals.upper_bound(right_open(4, 7));
        assert(it != intervals.end());
        assert(it->first.lower == 10);
        assert(it->first.upper == 12);
      }

      // const
      {
        auto it = std::as_const(intervals).upper_bound(right_open(4, 7));
        assert(it != intervals.end());
        assert(it->first.lower == 10);
        assert(it->first.upper == 12);
      }
    }

    // lower_bound
    {
      std::map<Interval, X, ExclusiveLess> intervals = {
          {right_open(0, 2), x},
          {right_open(3, 5), x},
          {right_open(6, 9), x},
          {right_open(10, 12), x},
      };
      // non-const
      {
        auto it = intervals.lower_bound(right_open(1, 4));
        assert(it != intervals.end());
        assert(it->first.lower == 0);
        assert(it->first.upper == 2);
      }

      // const
      {
        auto it = std::as_const(intervals).lower_bound(right_open(1, 4));
        assert(it != intervals.end());
        assert(it->first.lower == 0);
        assert(it->first.upper == 2);
      }
    }
  }
}
