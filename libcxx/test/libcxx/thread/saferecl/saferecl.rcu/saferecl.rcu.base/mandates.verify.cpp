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

#include <rcu>
#include <type_traits>

// D shall be a function object type ([function.objects]) for which, given a value d of type D and a value ptr of type T*, the expression d(ptr) is valid.
struct A {};
struct B : std::rcu_obj_base<B, A> {};
// expected-error-re@*:* {{static assertion failed {{.*}}Deleter must be callable with an object pointer.}}

// D shall meet the requirements for Cpp17DefaultConstructible and Cpp17MoveAssignable.
struct NonDefault {
  NonDefault(int) {}
  void operator()(auto...) const {}
};
struct C : std::rcu_obj_base<C, NonDefault> {};
// expected-error-re@*:* {{static assertion failed {{.*}}std::is_default_constructible_v<NonDefault>}}

struct NonMoveAssignable {
  NonMoveAssignable()                                = default;
  NonMoveAssignable(NonMoveAssignable&&)             = default;
  NonMoveAssignable& operator=(NonMoveAssignable&&) = delete;
};
struct D : std::rcu_obj_base<D, NonMoveAssignable> {};
// expected-error-re@*:* {{static assertion failed {{.*}}std::is_move_assignable_v<NonMoveAssignable>}}

// void retire(D d = D(), rcu_domain& dom = rcu_default_domain()) noexcept;
// Mandates: T is an rcu-protectable type.
struct E : std::rcu_obj_base<A> {};
void test(E* e) {
  e->retire();
  // expected-error-re@*:* {{static assertion failed {{.*}}T must be an rcu-protectable type.}}
  // expected-error-re@*:* {{static_cast {{.*}}}}
}
