//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <tuple>

// inline constexpr ignore-type ignore;

// std::ignore should be provided by the headers <tuple> and <utility>.
// This test validates its presence in <tuple>.

#include <tuple>

[[maybe_unused]] auto& ignore_v = std::ignore;
