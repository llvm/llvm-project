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

// <hazard_pointer>

// [[nodiscard]] is applied as a libc++ extension to hazard_pointer::empty(), protect(), try_protect() and
// make_hazard_pointer(): discarding these results is almost certainly a bug.

#include <hazard_pointer>
#include <atomic>

struct Node : std::hazard_pointer_obj_base<Node> {};

void test() {
  std::hazard_pointer h;
  std::atomic<Node*> src{nullptr};
  Node* p = nullptr;
  h.empty();                  // expected-warning {{ignoring return value of function}}
  h.protect(src);             // expected-warning {{ignoring return value of function}}
  h.try_protect(p, src);      // expected-warning {{ignoring return value of function}}
  std::make_hazard_pointer(); // expected-warning {{ignoring return value of function}}
}
