//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <cstddef>

// enum class byte : unsigned char {};

#include <cstddef>
#include <type_traits>

static_assert(std::is_enum_v<std::byte>);
static_assert(std::is_same_v<std::underlying_type_t<std::byte>, unsigned char>);
static_assert(!std::is_convertible_v<std::byte, unsigned char>);
