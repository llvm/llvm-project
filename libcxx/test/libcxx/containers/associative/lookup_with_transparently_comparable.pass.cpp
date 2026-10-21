//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Make sure that lookup operations on ordered associative containers are transparent
// when they can, even when not using a comparator that advertises itself as transparent
// (e.g. std::less<>). This is a libc++ QoI extension.
//
// At the moment, we only implement this optimization when the key stored in the container
// is std::string, so the tests are written with that assumption in mind.

#include <cassert>
#include <functional>
#include <map>
#include <set>
#include <string>

#include "count_new.h"
#include "test_macros.h"

template <class Container>
void test() {
  // Use a string longer than the SSO so that construction must allocate.
  const char* key = "long-string-to-exceed-SSO-buffer";

  // find
  {
    Container c;
    globalMemCounter.reset();
    (void)c.find(key);
    assert(globalMemCounter.checkNewCalledEq(0));
  }
  {
    Container const c;
    globalMemCounter.reset();
    (void)c.find(key);
    assert(globalMemCounter.checkNewCalledEq(0));
  }

  // count -- TODO: implement the optimization
  // {
  //   globalMemCounter.reset();
  //   (void)c.count(key);
  //   assert(globalMemCounter.checkNewCalledEq(0));
  // }

#if TEST_STD_VER >= 20
  // contains
  {
    Container c;
    globalMemCounter.reset();
    (void)c.contains(key);
    assert(globalMemCounter.checkNewCalledEq(0));
  }
  {
    Container const c;
    globalMemCounter.reset();
    (void)c.contains(key);
    assert(globalMemCounter.checkNewCalledEq(0));
  }
#endif

  // TODO: Implement the optimization for these methods
  // lower_bound
  // {
  //   Container c;
  //   globalMemCounter.reset();
  //   (void)c.lower_bound(key);
  //   assert(globalMemCounter.checkNewCalledEq(0));
  // }
  // {
  //   Container const c;
  //   globalMemCounter.reset();
  //   (void)c.lower_bound(key);
  //   assert(globalMemCounter.checkNewCalledEq(0));
  // }

  // upper_bound
  // {
  //   Container c;
  //   globalMemCounter.reset();
  //   (void)c.upper_bound(key);
  //   assert(globalMemCounter.checkNewCalledEq(0));
  // }
  // {
  //   Container const c;
  //   globalMemCounter.reset();
  //   (void)c.upper_bound(key);
  //   assert(globalMemCounter.checkNewCalledEq(0));
  // }

  // equal_range
  // {
  //   Container c;
  //   globalMemCounter.reset();
  //   (void)c.equal_range(key);
  //   assert(globalMemCounter.checkNewCalledEq(0));
  // }
  // {
  //   Container const c;
  //   globalMemCounter.reset();
  //   (void)c.equal_range(key);
  //   assert(globalMemCounter.checkNewCalledEq(0));
  // }
}

int main(int, char**) {
  test<std::map<std::string, int>>();
  // TODO: Implement the optimization for multimap, set and multiset
  // test<std::multimap<std::string, int>>();
  // test<std::set<std::string>>();
  // test<std::multiset<std::string>>();

  test<std::map<std::string, int, std::greater<std::string>>>();
  // test<std::multimap<std::string, int, std::greater<std::string>>>();
  // test<std::set<std::string, std::greater<std::string>>>();
  // test<std::multiset<std::string, std::greater<std::string>>>();

  {
    // std::map only: map::at
    std::map<std::string, int> c;
    char const* key = "long-string-to-exceed-SSO-buffer";
    c[key]          = 1;
    {
      globalMemCounter.reset();
      (void)c.at(key);
      assert(globalMemCounter.checkNewCalledEq(0));
    }
    {
      const std::map<std::string, int>& cc = c;
      globalMemCounter.reset();
      (void)cc.at(key);
      assert(globalMemCounter.checkNewCalledEq(0));
    }
  }

  return 0;
}
