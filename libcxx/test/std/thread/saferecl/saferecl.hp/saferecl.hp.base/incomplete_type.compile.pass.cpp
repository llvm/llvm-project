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

// [saferecl.hp.base]/4: T may be an incomplete type. It shall be complete before any member of the
// resulting specialization of hazard_pointer_obj_base is referenced.

#include <hazard_pointer>

#include "test_macros.h"

struct Incomplete;
std::hazard_pointer_obj_base<Incomplete>* declared_only(); // the specialization is named while T is incomplete

struct Node : std::hazard_pointer_obj_base<Node> { // T is incomplete while the base is instantiated
  Node* next = nullptr;
};

struct Deleter;
struct WithForwardDeclaredT;
struct Deleter {
  void operator()(WithForwardDeclaredT*) const noexcept;
};
struct WithForwardDeclaredT : std::hazard_pointer_obj_base<WithForwardDeclaredT, Deleter> {};

void use() {
  Node n;
  n.retire(); // T is complete here
}
