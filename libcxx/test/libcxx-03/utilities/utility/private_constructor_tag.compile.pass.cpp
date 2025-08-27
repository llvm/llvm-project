//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// struct __private_constructor_tag{};

// The private constructor tag is intended to be a trivial type that can easily
// be used to mark a constructor exposition-only.
//
// Tests whether the type is trivial.

#include <__cxx03/__utility/private_constructor_tag.h>
#include <type_traits>

static_assert(std::is_trivially_copyable<std::__private_constructor_tag>::value, "");
static_assert(std::is_trivially_default_constructible<std::__private_constructor_tag>::value, "");
