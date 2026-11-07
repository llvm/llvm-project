//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// ~hive() is not constexpr, as in [hive.overview]: a hive can be constant-initialized, but cannot be created and
// destroyed during constant evaluation.

#include <hive>
#include <type_traits>

constinit std::hive<int> a;
constinit std::hive<int> b(std::hive_limits(64, 128));

// Whether creating and destroying a T is a constant expression.
template <class T>
concept ConstantDestructible = requires { typename std::bool_constant<(static_cast<void>(T()), true)>; };

struct ConstexprDestructor {
  constexpr ConstexprDestructor() {}
  constexpr ~ConstexprDestructor() {}
};

struct RuntimeDestructor {
  constexpr RuntimeDestructor() {}
  ~RuntimeDestructor() {}
};

static_assert(ConstantDestructible<ConstexprDestructor>);
static_assert(!ConstantDestructible<RuntimeDestructor>);
static_assert(ConstantDestructible<std::hive<int>::iterator>);
static_assert(!ConstantDestructible<std::hive<int>>);
