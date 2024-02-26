//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>

// Ordinary unqualified lookup should not be performed.

namespace ns {
struct StructWithGlobalCmpFunctions {};
} // namespace ns

struct ConvertibleToCmpType;

ConvertibleToCmpType strong_order(ns::StructWithGlobalCmpFunctions, ns::StructWithGlobalCmpFunctions);
ConvertibleToCmpType weak_order(ns::StructWithGlobalCmpFunctions, ns::StructWithGlobalCmpFunctions);
ConvertibleToCmpType partial_order(ns::StructWithGlobalCmpFunctions, ns::StructWithGlobalCmpFunctions);

#include <compare>
#include <type_traits>

struct ConvertibleToCmpType {
  operator std::strong_ordering() const;
  operator std::weak_ordering() const;
  operator std::partial_ordering() const;
};

static_assert(!std::is_invocable_v<decltype(std::strong_order),
                                   ns::StructWithGlobalCmpFunctions,
                                   ns::StructWithGlobalCmpFunctions>);

static_assert(!std::is_invocable_v<decltype(std::weak_order),
                                   ns::StructWithGlobalCmpFunctions,
                                   ns::StructWithGlobalCmpFunctions>);

static_assert(!std::is_invocable_v<decltype(std::partial_order),
                                   ns::StructWithGlobalCmpFunctions,
                                   ns::StructWithGlobalCmpFunctions>);
