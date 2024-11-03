//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://github.com/llvm/llvm-project/issues/40340 is fixed
// UNSUPPORTED: availability-pmr-missing

// <memory_resource>

// memory_resource::do_allocate(size_t, size_t);          /* private */
// memory_resource::do_deallocate(void*, size_t, size_t); /* private */
// memory_resource::do_is_equal(memory_resource const&);  /* private */

#include <memory_resource>

void test() {
  std::pmr::memory_resource* m = std::pmr::new_delete_resource();
  m->do_allocate(0, 0);            // expected-error{{'do_allocate' is a private member}}
  m->do_deallocate(nullptr, 0, 0); // expected-error{{'do_deallocate' is a private member}}
  m->do_is_equal(*m);              // expected-error{{'do_is_equal' is a private member}}
}
