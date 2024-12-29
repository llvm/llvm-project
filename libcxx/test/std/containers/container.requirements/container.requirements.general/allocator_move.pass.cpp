//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// C++2a[container.requirements.general]p8
//   Move constructors obtain an allocator by move construction from the allocator
//   belonging to the container being moved. Such move construction of the
//   allocator shall not exit via an exception.

#include <vector>
#include <deque>
#include <list>
#include <forward_list>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "test_macros.h"
#include "test_allocator.h"

template <class C>
void test(int expected_num_allocs = 1) {
  test_allocator_statistics alloc_stats;
  {
    alloc_stats.clear();
    using AllocT = typename C::allocator_type;
    C v(AllocT(42, 101, &alloc_stats));

    assert(alloc_stats.count == expected_num_allocs);

    const int num_stored_allocs = alloc_stats.count;
    {
      const AllocT& a = v.get_allocator();
      assert(alloc_stats.count == 1 + num_stored_allocs);
      assert(a.get_data() == 42);
      assert(a.get_id() == 101);
    }
    assert(alloc_stats.count == num_stored_allocs);
    alloc_stats.clear_ctor_counters();

    C v2 = std::move(v);
    assert(alloc_stats.count == num_stored_allocs * 2);
    assert(alloc_stats.copied == 0);
    assert(alloc_stats.moved == num_stored_allocs);
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
}

int main(int, char**) {
  { // test sequence containers
    test<std::vector<int, test_allocator<int> > >();
    test<std::vector<bool, test_allocator<bool> > >();
    test<std::list<int, test_allocator<int> > >();
    test<std::forward_list<int, test_allocator<int> > >();

    // libc++ stores two allocators in deque
#ifdef _LIBCPP_VERSION
    int stored_allocators = 2;
#else
    int stored_allocators = 1;
#endif
    test<std::deque<int, test_allocator<int> > >(stored_allocators);
  }
  { // test associative containers
    test<std::set<int, std::less<int>, test_allocator<int> > >();
    test<std::multiset<int, std::less<int>, test_allocator<int> > >();

    using KV = std::pair<const int, int>;
    test<std::map<int, int, std::less<int>, test_allocator<KV> > >();
    test<std::multimap<int, int, std::less<int>, test_allocator<KV> > >();
  }
  { // test unordered containers
    // libc++ stores two allocators in the unordered containers.
#ifdef _LIBCPP_VERSION
    int stored_allocators = 2;
#else
    int stored_allocators = 1;
#endif
    test<std::unordered_set<int, std::hash<int>, std::equal_to<int>,
                            test_allocator<int> > >(stored_allocators);
    test<std::unordered_multiset<int, std::hash<int>, std::equal_to<int>,
                                 test_allocator<int> > >(stored_allocators);

    using KV = std::pair<const int, int>;
    test<std::unordered_map<int, int, std::hash<int>, std::equal_to<int>,
                            test_allocator<KV> > >(stored_allocators);
    test<std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>,
                                 test_allocator<KV> > >(stored_allocators);
  }

  return 0;
}
