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

// template<class T, class D = default_delete<T>>
//   class rcu_obj_base
// T may be an incomplete type. It shall be complete before any member of the resulting specialization of rcu_obj_base is referenced.

#include <rcu>
#include <type_traits>

class A;

void test(std::rcu_obj_base<A>*) {}
