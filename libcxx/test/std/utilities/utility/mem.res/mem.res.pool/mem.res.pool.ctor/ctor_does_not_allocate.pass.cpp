//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// <memory_resource>

// class synchronized_pool_resource
// class unsynchronized_pool_resource

#include <memory_resource>
#include <cassert>

#include "count_new.h"

template <class PoolResource>
void test() {
  // Constructing a pool resource should not cause allocations
  // by itself; the resource should wait to allocate until an
  // allocation is requested.

  globalMemCounter.reset();
  std::pmr::set_default_resource(std::pmr::new_delete_resource());

  PoolResource r1;
  assert(globalMemCounter.checkNewCalledEq(0));

  PoolResource r2(std::pmr::pool_options{1024, 2048});
  assert(globalMemCounter.checkNewCalledEq(0));

  PoolResource r3(std::pmr::pool_options{1024, 2048}, std::pmr::new_delete_resource());
  assert(globalMemCounter.checkNewCalledEq(0));
}

int main(int, char**) {
  test<std::pmr::unsynchronized_pool_resource>();
  test<std::pmr::synchronized_pool_resource>();

  return 0;
}
