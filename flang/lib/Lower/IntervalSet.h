//===-- IntervalSet.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_INTERVALSET_H
#define FORTRAN_LOWER_INTERVALSET_H

#include <cassert>
#include <map>

namespace Fortran::lower {

//===----------------------------------------------------------------------===//
// Interval set
//===----------------------------------------------------------------------===//

/// Interval set to keep track of intervals, merging them when they overlap one
/// another. Used to refine the pseudo-offset ranges of the front-end symbols
/// into groups of aliasing variables.
struct IntervalSet {
  using MAP = std::map<std::size_t, std::size_t>;
  using Iterator = MAP::const_iterator;

  // Handles the merging of overlapping intervals correctly, efficiently.
  void merge(std::size_t lo, std::size_t up) {
    assert(lo <= up);
    if (empty()) {
      m.insert({lo, up});
      return;
    }
    auto i = m.lower_bound(lo);
    // i->first >= lo
    if (i == begin()) {
      if (up < i->first) {
        // [lo..up] < i->first
        m.insert({lo, up});
        return;
      }
      // up >= i->first
      if (i->second > up)
        up = i->second;
      fuse(lo, up, i);
      return;
    }
    auto i1 = i;
    if (i == end() || i->first > lo)
      i = std::prev(i);
    // i->first <= lo
    if (i->second >= up) {
      // i->first <= lo && up <= i->second, keep i
      return;
    }
    // i->second < up
    if (i->second < lo) {
      if (i1 == end() || i1->first > up) {
        // i < [lo..up] < i1
        m.insert({lo, up});
        return;
      }
      // i < [lo..up], i1->first <= up  -->  [lo..up] union [i1..?]
      i = i1;
    } else {
      // i->first <= lo, lo <= i->second  -->  [i->first..up] union [i..?]
      lo = i->first;
    }
    fuse(lo, up, i);
  }

  Iterator find(std::size_t pt) const {
    auto i = m.lower_bound(pt);
    if (i != end() && i->first == pt)
      return i;
    if (i == begin())
      return end();
    i = std::prev(i);
    if (i->second < pt)
      return end();
    return i;
  }

  Iterator begin() const { return m.begin(); }
  Iterator end() const { return m.end(); }
  bool empty() const { return m.empty(); }
  std::size_t size() const { return m.size(); }

private:
  // Find and fuse overlapping sets.
  void fuse(std::size_t lo, std::size_t up, Iterator i) {
    auto j = m.upper_bound(up);
    // up < j->first
    auto cu = std::prev(j)->second;
    // cu < j->first
    if (cu > up)
      up = cu;
    m.erase(i, j);
    // merge [i .. j) with [i->first, max(up, cu)]
    m.insert({lo, up});
  }

  MAP m{};
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_INTERVALSET_H
