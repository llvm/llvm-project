//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-debug-mode, c++03, c++11, c++14

// test container debugging

#include <string>
#include <vector>

#include "test_macros.h"
#include "check_assertion.h"
#include "container_debug_tests.h"

using namespace IteratorDebugChecks;

typedef std::basic_string<char, std::char_traits<char>, test_allocator<char>>  StringType;

template <class Container = StringType, ContainerType CT = CT_String>
struct StringContainerChecks : BasicContainerChecks<Container, CT> {
  using Base = BasicContainerChecks<Container, CT_String>;
  using value_type = typename Container::value_type;
  using allocator_type = typename Container::allocator_type;
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;

  using Base::makeContainer;
  using Base::makeValueType;

public:
  static void run() {
    Base::run_iterator_tests();
    Base::run_allocator_aware_tests();

    for (int N : {3, 128}) {
      FrontOnEmptyContainer(N);
      BackOnEmptyContainer(N);
      PopBack(N);
    }
  }

private:
  static void BackOnEmptyContainer(int N) {
    // testing back on empty
    Container C = makeContainer(N);
    Container const& CC = C;
    iterator it = --C.end();
    (void)C.back();
    (void)CC.back();
    C.pop_back();
    EXPECT_DEATH( C.erase(it) );
    C.clear();
    EXPECT_DEATH( C.back() );
    EXPECT_DEATH( CC.back() );
  }

  static void FrontOnEmptyContainer(int N) {
    // testing front on empty
    Container C = makeContainer(N);
    Container const& CC = C;
    (void)C.front();
    (void)CC.front();
    C.clear();
    EXPECT_DEATH( C.front() );
    EXPECT_DEATH( CC.front() );
  }

  static void PopBack(int N) {
    // testing pop_back() invalidation
    Container C1 = makeContainer(N);
    iterator it1 = C1.end();
    --it1;
    C1.pop_back();
    EXPECT_DEATH( C1.erase(it1) );
    C1.erase(C1.begin(), C1.end());
    assert(C1.size() == 0);
    TEST_LIBCPP_ASSERT_FAILURE(C1.pop_back(), "string::pop_back(): string is already empty");
  }
};

int main(int, char**)
{
  StringContainerChecks<>::run();

  return 0;
}
