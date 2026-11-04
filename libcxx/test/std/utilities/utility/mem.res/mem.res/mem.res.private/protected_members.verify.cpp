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

// monotonic_buffer_resource::do_allocate(size_t, size_t);          /* protected */
// monotonic_buffer_resource::do_deallocate(void*, size_t, size_t); /* protected */
// monotonic_buffer_resource::do_is_equal(memory_resource const&);  /* protected */

// synchronized_pool_resource::do_allocate(size_t, size_t);          /* protected */
// synchronized_pool_resource::do_deallocate(void*, size_t, size_t); /* protected */
// synchronized_pool_resource::do_is_equal(memory_resource const&);  /* protected */

// unsynchronized_pool_resource::do_allocate(size_t, size_t);          /* protected */
// unsynchronized_pool_resource::do_deallocate(void*, size_t, size_t); /* protected */
// unsynchronized_pool_resource::do_is_equal(memory_resource const&);  /* protected */

#include <memory_resource>

void test() {
  {
    std::pmr::monotonic_buffer_resource m;
    m.do_allocate(0, 0);            // expected-error{{'do_allocate' is a protected member}}
    m.do_deallocate(nullptr, 0, 0); // expected-error{{'do_deallocate' is a protected member}}
    m.do_is_equal(m);               // expected-error{{'do_is_equal' is a protected member}}
  }
  {
    std::pmr::synchronized_pool_resource m;
    m.do_allocate(0, 0);            // expected-error{{'do_allocate' is a protected member}}
    m.do_deallocate(nullptr, 0, 0); // expected-error{{'do_deallocate' is a protected member}}
    m.do_is_equal(m);               // expected-error{{'do_is_equal' is a protected member}}
  }
  {
    std::pmr::unsynchronized_pool_resource m;
    m.do_allocate(0, 0);            // expected-error{{'do_allocate' is a protected member}}
    m.do_deallocate(nullptr, 0, 0); // expected-error{{'do_deallocate' is a protected member}}
    m.do_is_equal(m);               // expected-error{{'do_is_equal' is a protected member}}
  }
}
