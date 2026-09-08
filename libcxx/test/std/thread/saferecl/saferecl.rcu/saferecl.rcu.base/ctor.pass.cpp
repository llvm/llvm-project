//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// REQUIRES: std-at-least-c++26

// protected:
//  rcu_obj_base() = default;
//  rcu_obj_base(const rcu_obj_base&) = default;
//  rcu_obj_base(rcu_obj_base&&) = default;
//  rcu_obj_base& operator=(const rcu_obj_base&) = default;
//  rcu_obj_base& operator=(rcu_obj_base&&) = default;
//  ~rcu_obj_base() = default;

#include <cassert>
#include <rcu>
#include <thread>
#include <type_traits>

#include "make_test_thread.h"
#include "test_macros.h"

class A : public std::rcu_obj_base<A> {};

static_assert(!std::is_default_constructible_v<std::rcu_obj_base<A>>);
static_assert(!std::is_copy_constructible_v<std::rcu_obj_base<A>>);
static_assert(!std::is_move_constructible_v<std::rcu_obj_base<A>>);
static_assert(!std::is_copy_assignable_v<std::rcu_obj_base<A>>);
static_assert(!std::is_move_assignable_v<std::rcu_obj_base<A>>);
static_assert(!std::is_destructible_v<std::rcu_obj_base<A>>);

static_assert(std::is_default_constructible_v<A>);
static_assert(std::is_copy_constructible_v<A>);
static_assert(std::is_move_constructible_v<A>);
static_assert(std::is_copy_assignable_v<A>);
static_assert(std::is_move_assignable_v<A>);
static_assert(std::is_destructible_v<A>);

// If D is trivially copyable, all specializations of rcu_obj_base<T, D> are trivially copyable.
struct TrivialDeleter {
  void operator()(auto...) const {}
};
struct B : std::rcu_obj_base<B, TrivialDeleter> {};
static_assert(std::is_trivially_copyable_v<std::rcu_obj_base<A, TrivialDeleter>>);

int main(int, char**) {
  {
    A a1;
    A a2(a1);
    A a3(std::move(a1));
    a1 = a2;
    a1 = std::move(a3);
  }

  return 0;
}
