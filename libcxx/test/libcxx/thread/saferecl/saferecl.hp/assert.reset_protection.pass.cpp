//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-threads
// XFAIL: availability-hazard_pointer-missing

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <hazard_pointer>
// template<class T> void reset_protection(const T* ptr) noexcept;
// void reset_protection(nullptr_t = nullptr) noexcept;              Preconditions: *this is not empty.

#include <hazard_pointer>

#include "check_assertion.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

int main(int, char**) {
  std::hazard_pointer empty;
  Node node;
  TEST_LIBCPP_ASSERT_FAILURE(
      empty.reset_protection(&node), "hazard_pointer::reset_protection(): hazard_pointer is empty");
  TEST_LIBCPP_ASSERT_FAILURE(empty.reset_protection(), "hazard_pointer::reset_protection(): hazard_pointer is empty");
  TEST_LIBCPP_ASSERT_FAILURE(
      empty.reset_protection(nullptr), "hazard_pointer::reset_protection(): hazard_pointer is empty");
  return 0;
}
