//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that std::__unwrap_iter() returns the correct type

#include <algorithm>
#include <cassert>
#include <string>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
using UnwrapT = decltype(std::__unwrap_iter(std::declval<Iter>()));

template <class Iter>
using rev_iter = std::reverse_iterator<Iter>;

template <class Iter>
using rev_rev_iter = rev_iter<rev_iter<Iter> >;

static_assert(std::is_same<UnwrapT<int*>, int*>::value, "");
static_assert(std::is_same<UnwrapT<std::__wrap_iter<int*> >, int*>::value, "");
static_assert(std::is_same<UnwrapT<rev_iter<int*> >, std::reverse_iterator<int*> >::value, "");
static_assert(std::is_same<UnwrapT<rev_rev_iter<int*> >, int*>::value, "");
static_assert(std::is_same<UnwrapT<rev_rev_iter<std::__wrap_iter<int*> > >, int*>::value, "");
static_assert(std::is_same<UnwrapT<rev_rev_iter<rev_iter<std::__wrap_iter<int*> > > >, rev_iter<std::__wrap_iter<int*> > >::value, "");

static_assert(std::is_same<UnwrapT<random_access_iterator<int*> >, random_access_iterator<int*> >::value, "");
static_assert(std::is_same<UnwrapT<rev_iter<random_access_iterator<int*> > >, rev_iter<random_access_iterator<int*> > >::value, "");
static_assert(std::is_same<UnwrapT<rev_rev_iter<random_access_iterator<int*> > >, random_access_iterator<int*> >::value, "");
static_assert(std::is_same<UnwrapT<rev_rev_iter<rev_iter<random_access_iterator<int*> > > >, rev_iter<random_access_iterator<int*> > >::value, "");

TEST_CONSTEXPR_CXX20 bool test() {
  std::string str = "Banane";
  using Iter = std::string::iterator;

  assert(std::__unwrap_iter(str.begin()) == str.data());
  assert(std::__unwrap_iter(str.end()) == str.data() + str.size());
  assert(std::__unwrap_iter(rev_rev_iter<Iter>(rev_iter<Iter>(str.begin()))) == str.data());
  assert(std::__unwrap_iter(rev_rev_iter<Iter>(rev_iter<Iter>(str.end()))) == str.data() + str.size());

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
