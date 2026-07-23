//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Pin down the ABI of associative containers with respect to their size and alignment
// when passed a comparator that is a reference.
//
// While it's not even clear that reference comparators are legal in containers, an
// unintended ABI break was discovered after implementing the new compressed pair
// mechanism based on [[no_unique_address]], and this is a regression test for that.
// If we decide to make reference comparators ill-formed, this test would become
// unnecessary.
//
// See https://llvm.org/PR118559 for more details.

#include <set>
#include <map>

#include "test_macros.h"

struct TEST_ALIGNAS(16) Cmp {
  bool operator()(int, int) const;
};

template <class Compare>
struct Set {
  char b;
  std::set<int, Compare> s;
};

template <class Compare>
struct Multiset {
  char b;
  std::multiset<int, Compare> s;
};

template <class Compare>
struct Map {
  char b;
  std::map<int, char, Compare> s;
};

template <class Compare>
struct Multimap {
  char b;
  std::multimap<int, char, Compare> s;
};

static_assert(sizeof(Set<Cmp&>) == sizeof(Set<bool (*)(int, int)>), "");
static_assert(sizeof(Multiset<Cmp&>) == sizeof(Multiset<bool (*)(int, int)>), "");

static_assert(sizeof(Map<Cmp&>) == sizeof(Map<bool (*)(int, int)>), "");
static_assert(sizeof(Multimap<Cmp&>) == sizeof(Multimap<bool (*)(int, int)>), "");
