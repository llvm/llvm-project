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
// void retire(D d = D()) noexcept;   Preconditions: *this is not retired.

#include <hazard_pointer>

#include "check_assertion.h"

struct Noop {
  void operator()(struct Node*) const noexcept {}
};
struct Node : std::hazard_pointer_obj_base<Node, Noop> {};

int main(int, char**) {
  Node node; // never reclaimed with a real deleter, so retiring it twice is observable and harmless here
  node.retire();
  TEST_LIBCPP_ASSERT_FAILURE(node.retire(), "hazard_pointer_obj_base::retire(): object has already been retired");
  return 0;
}
