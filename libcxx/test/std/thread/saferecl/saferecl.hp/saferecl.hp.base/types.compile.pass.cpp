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

// template<class T, class D = default_delete<T>> class hazard_pointer_obj_base;
//   void retire(D d = D()) noexcept;
//   protected: defaulted special members

#include <hazard_pointer>
#include <memory>
#include <type_traits>
#include <utility>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

struct Deleter {
  void operator()(struct WithDeleter*) const noexcept {}
};
struct WithDeleter : std::hazard_pointer_obj_base<WithDeleter, Deleter> {};

// The default deleter is default_delete<T>.
static_assert(std::is_base_of_v<std::hazard_pointer_obj_base<Node, std::default_delete<Node>>, Node>);

// retire is noexcept and takes D by value with a default argument.
static_assert(noexcept(std::declval<Node&>().retire()));
static_assert(noexcept(std::declval<Node&>().retire(std::default_delete<Node>())));
static_assert(noexcept(std::declval<WithDeleter&>().retire(Deleter())));
static_assert(std::is_same_v<decltype(std::declval<Node&>().retire()), void>);

// The special members are protected: not usable from outside, but usable by the derived class.
static_assert(!std::is_default_constructible_v<std::hazard_pointer_obj_base<Node>>);
static_assert(!std::is_copy_constructible_v<std::hazard_pointer_obj_base<Node>>);
static_assert(!std::is_destructible_v<std::hazard_pointer_obj_base<Node>>);
static_assert(std::is_default_constructible_v<Node>);
static_assert(std::is_copy_constructible_v<Node>);
static_assert(std::is_move_constructible_v<Node>);
static_assert(std::is_copy_assignable_v<Node>);
static_assert(std::is_move_assignable_v<Node>);
static_assert(std::is_destructible_v<Node>);
