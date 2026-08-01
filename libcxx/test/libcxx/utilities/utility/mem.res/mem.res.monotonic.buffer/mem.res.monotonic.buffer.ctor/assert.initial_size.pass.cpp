//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory_resource>

// class monotonic_buffer_resource

// explicit monotonic_buffer_resource(size_t initial_size);
// monotonic_buffer_resource(size_t initial_size, memory_resource* upstream);

// [mem.res.monotonic.buffer.ctor] requires initial_size to be greater than zero.
// Make sure that passing zero triggers an assertion.

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: availability-pmr-missing
// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <cstddef>
#include <memory_resource>

#include "check_assertion.h"

int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(std::pmr::monotonic_buffer_resource(std::size_t(0)),
                             "monotonic_buffer_resource: initial_size must be greater than zero");
  TEST_LIBCPP_ASSERT_FAILURE(std::pmr::monotonic_buffer_resource(std::size_t(0), std::pmr::new_delete_resource()),
                             "monotonic_buffer_resource: initial_size must be greater than zero");

  return 0;
}
