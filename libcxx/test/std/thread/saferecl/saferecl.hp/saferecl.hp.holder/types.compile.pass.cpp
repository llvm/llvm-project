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

// class hazard_pointer;
//   The class is move-only and every member listed in [saferecl.hp.holder] is present with the return
//   type and noexcept-ness the synopsis gives: the default constructor, the move constructor, move
//   assignment, empty(), protect(), try_protect(), both reset_protection() overloads, swap(), the free
//   swap() and make_hazard_pointer().

#include <hazard_pointer>
#include <atomic>
#include <type_traits>
#include <utility>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

static_assert(std::is_nothrow_default_constructible_v<std::hazard_pointer>);
static_assert(std::is_nothrow_move_constructible_v<std::hazard_pointer>);
static_assert(std::is_nothrow_move_assignable_v<std::hazard_pointer>);
static_assert(!std::is_copy_constructible_v<std::hazard_pointer>);
static_assert(!std::is_copy_assignable_v<std::hazard_pointer>);
static_assert(std::is_nothrow_swappable_v<std::hazard_pointer>);
static_assert(noexcept(std::declval<const std::hazard_pointer&>().empty()));
static_assert(noexcept(std::declval<std::hazard_pointer&>().protect(std::declval<const std::atomic<Node*>&>())));
static_assert(noexcept(std::declval<std::hazard_pointer&>().try_protect(
    std::declval<Node*&>(), std::declval<const std::atomic<Node*>&>())));
static_assert(noexcept(std::declval<std::hazard_pointer&>().reset_protection(std::declval<const Node*>())));
static_assert(noexcept(std::declval<std::hazard_pointer&>().reset_protection()));
static_assert(noexcept(std::declval<std::hazard_pointer&>().reset_protection(nullptr)));
static_assert(noexcept(std::declval<std::hazard_pointer&>().swap(std::declval<std::hazard_pointer&>())));
static_assert(noexcept(std::swap(std::declval<std::hazard_pointer&>(), std::declval<std::hazard_pointer&>())));
static_assert(!noexcept(std::make_hazard_pointer()));
static_assert(std::is_same_v<decltype(std::make_hazard_pointer()), std::hazard_pointer>);
static_assert(
    std::is_same_v<decltype(std::declval<std::hazard_pointer&>().protect(std::declval<const std::atomic<Node*>&>())),
                   Node*>);
static_assert(std::is_same_v<decltype(std::declval<std::hazard_pointer&>().try_protect(
                                 std::declval<Node*&>(), std::declval<const std::atomic<Node*>&>())),
                             bool>);
