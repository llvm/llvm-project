//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <format>

// enum class range_format {
//   disabled,
//   map,
//   set,
//   sequence,
//   string,
//   debug_string
// };

#include <format>

// test that the enumeration values exist
static_assert(requires { std::range_format::disabled; });
static_assert(requires { std::range_format::map; });
static_assert(requires { std::range_format::set; });
static_assert(requires { std::range_format::sequence; });
static_assert(requires { std::range_format::string; });
static_assert(requires { std::range_format::debug_string; });
