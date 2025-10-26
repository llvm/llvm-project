//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23, has-unix-headers, libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing
// XFAIL: availability-stacktrace-missing

// Hardened requirements for the `current` call with given `skip` and `max_depth` amounts:
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3697r0.html#basic_stacktrace
// Specifically: "Hardened preconditions: skip <= skip + max_depth is true."
//
// (19.6.4.2): [stacktrace.basic.cons], creation and assignment
// static basic_stacktrace current(size_type skip, size_type max_depth,
//                                 const allocator_type& alloc = allocator_type()) noexcept;

#include <stacktrace>

#include "check_assertion.h"
#include "test_macros.h"

int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(
      std::stacktrace::current(1, 0xffffffffffffffff), "sum of skip and max_depth overflows size_type");

  return 0;
}
