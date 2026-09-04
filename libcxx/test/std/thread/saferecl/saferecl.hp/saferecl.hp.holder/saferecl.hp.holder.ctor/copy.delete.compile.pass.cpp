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

// hazard_pointer(const hazard_pointer&) = delete;   (implicitly, since move members are declared)
// hazard_pointer& operator=(const hazard_pointer&) = delete;

#include <hazard_pointer>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<std::hazard_pointer>);
static_assert(!std::is_copy_assignable_v<std::hazard_pointer>);
static_assert(std::is_nothrow_move_constructible_v<std::hazard_pointer>);
static_assert(std::is_nothrow_move_assignable_v<std::hazard_pointer>);
