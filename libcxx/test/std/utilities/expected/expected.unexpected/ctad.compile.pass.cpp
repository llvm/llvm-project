//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class E> unexpected(E) -> unexpected<E>;

#include <concepts>
#include <expected>

struct Foo{};

static_assert(std::same_as<decltype(std::unexpected(5)), std::unexpected<int>>);
static_assert(std::same_as<decltype(std::unexpected(Foo{})), std::unexpected<Foo>>);
static_assert(std::same_as<decltype(std::unexpected(std::unexpected<int>(5))), std::unexpected<int>>);
