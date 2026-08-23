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

// Ensure that we never change the size or alignment of the ABI-visible hazard pointer types.
// hazard_pointer is one pointer to a slot; the per-object node is {next, reclaim}; the deleter of
// hazard_pointer_obj_base uses [[no_unique_address]] so an empty deleter costs nothing.

#include <hazard_pointer>
#include <memory>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

struct StatefulDeleter {
  void* state;
  void operator()(struct Big*) const noexcept {}
};
struct Big : std::hazard_pointer_obj_base<Big, StatefulDeleter> {};

static_assert(sizeof(std::hazard_pointer) == sizeof(void*));
static_assert(alignof(std::hazard_pointer) == alignof(void*));

static_assert(sizeof(std::__hazard_pointer_slot) == sizeof(void*));
static_assert(alignof(std::__hazard_pointer_slot) == alignof(void*));

static_assert(sizeof(std::__hazard_pointer_obj_node) == 2 * sizeof(void*));
static_assert(alignof(std::__hazard_pointer_obj_node) == alignof(void*));

static_assert(sizeof(std::hazard_pointer_obj_base<Node>) == 2 * sizeof(void*));
static_assert(alignof(std::hazard_pointer_obj_base<Node>) == alignof(void*));
static_assert(sizeof(Node) == 2 * sizeof(void*));

static_assert(sizeof(std::hazard_pointer_obj_base<Big, StatefulDeleter>) == 3 * sizeof(void*));
static_assert(alignof(std::hazard_pointer_obj_base<Big, StatefulDeleter>) == alignof(void*));
