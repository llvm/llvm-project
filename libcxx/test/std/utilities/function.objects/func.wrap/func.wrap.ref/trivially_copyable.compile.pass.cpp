//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// Each specialization of function_ref is a trivially copyable type ([basic.types.general]) that models copyable.

#include <concepts>
#include <functional>
#include <type_traits>

struct A {
  A() {}
  A(const A&) {}
  ~A() {}
};

static_assert(std::copyable<std::function_ref<A(const A&)>>);
static_assert(std::is_trivially_copyable_v<std::function_ref<A(const A&)>>);

static_assert(std::copyable<std::function_ref<A(const A&) noexcept>>);
static_assert(std::is_trivially_copyable_v<std::function_ref<A(const A&) noexcept>>);

static_assert(std::copyable<std::function_ref<A(const A&) const>>);
static_assert(std::is_trivially_copyable_v<std::function_ref<A(const A&) const>>);

static_assert(std::copyable<std::function_ref<A(const A&) const noexcept>>);
static_assert(std::is_trivially_copyable_v<std::function_ref<A(const A&) const noexcept>>);
