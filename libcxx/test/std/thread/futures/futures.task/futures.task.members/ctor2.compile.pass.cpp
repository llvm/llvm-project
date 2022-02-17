//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <future>

// class packaged_task<R(ArgTypes...)>
// template <class F, class Allocator>
//   packaged_task(allocator_arg_t, const Allocator& a, F&& f);
// These constructors shall not participate in overload resolution if
//    decay<F>::type is the same type as std::packaged_task<R(ArgTypes...)>.

#include <cassert>
#include <future>

#include "test_allocator.h"

struct A {};
using PT = std::packaged_task<A(int, char)>;
using VPT = volatile std::packaged_task<A(int, char)>;

static_assert(!std::is_constructible<PT, std::allocator_arg_t, test_allocator<A>, VPT>::value, "");

using PA = std::packaged_task<A(int)>;
using PI = std::packaged_task<int(int)>;

static_assert(!std::is_constructible<PA, std::allocator_arg_t, std::allocator<A>, const PA&>::value, "");
static_assert(!std::is_constructible<PA, std::allocator_arg_t, std::allocator<A>, const PA&&>::value, "");
static_assert(!std::is_constructible<PA, std::allocator_arg_t, std::allocator<A>, volatile PA&>::value, "");
static_assert(!std::is_constructible<PA, std::allocator_arg_t, std::allocator<A>, volatile PA&&>::value, "");

static_assert( std::is_constructible<PA, std::allocator_arg_t, std::allocator<A>, const PI&>::value, "");
static_assert( std::is_constructible<PA, std::allocator_arg_t, std::allocator<A>, const PI&&>::value, "");
static_assert( std::is_constructible<PA, std::allocator_arg_t, std::allocator<A>, volatile PI&>::value, "");
static_assert( std::is_constructible<PA, std::allocator_arg_t, std::allocator<A>, volatile PI&&>::value, "");
