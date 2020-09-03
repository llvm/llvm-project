//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: windows
// UNSUPPORTED: libcpp-no-if-constexpr
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

// Can't test the system lib because this test enables debug mode
// UNSUPPORTED: with_system_cxx_lib=macosx

// test container debugging

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <cassert>
#include "container_debug_tests.h"
#include "test_macros.h"
#include "debug_mode_helper.h"

using namespace IteratorDebugChecks;

template <class Container, ContainerType CT>
struct UnorderedContainerChecks : BasicContainerChecks<Container, CT> {
  using Base = BasicContainerChecks<Container, CT>;
  using value_type = typename Container::value_type;
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;
  using traits = std::iterator_traits<iterator>;
  using category = typename traits::iterator_category;

  using Base::makeContainer;
public:
  static void run() {
    Base::run();
  }
private:

};

int main(int, char**)
{
  using SetAlloc = test_allocator<int>;
  using MapAlloc = test_allocator<std::pair<const int, int>>;
  {
    UnorderedContainerChecks<
        std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, MapAlloc>,
        CT_UnorderedMap>::run();
    UnorderedContainerChecks<
        std::unordered_set<int, std::hash<int>, std::equal_to<int>, SetAlloc>,
        CT_UnorderedSet>::run();
    UnorderedContainerChecks<
        std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, MapAlloc>,
        CT_UnorderedMultiMap>::run();
    UnorderedContainerChecks<
        std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, SetAlloc>,
        CT_UnorderedMultiSet>::run();
  }

  return 0;
}
