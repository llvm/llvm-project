//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/memory_resource>

// memory_resource::do_allocate(size_t, size_t);          /* protected */
// memory_resource::do_deallocate(void*, size_t, size_t); /* protected */
// memory_resource::do_is_equal(memory_resource const&);  /* protected */

#include <experimental/memory_resource>

namespace ex = std::experimental::pmr;

int main(int, char**) {
    ex::memory_resource *m = ex::new_delete_resource();
    m->do_allocate(0, 0); // expected-error{{'do_allocate' is a protected member}}
    m->do_deallocate(nullptr, 0, 0); // expected-error{{'do_deallocate' is a protected member}}
    m->do_is_equal(*m); // expected-error{{'do_is_equal' is a protected member}}

  return 0;
}
