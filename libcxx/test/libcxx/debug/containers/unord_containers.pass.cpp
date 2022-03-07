//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14

// UNSUPPORTED: libcxx-no-debug-mode, c++03, windows
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

// test container debugging

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <cassert>
#include "check_assertion.h"
#include "container_debug_tests.h"
#include "test_macros.h"

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
