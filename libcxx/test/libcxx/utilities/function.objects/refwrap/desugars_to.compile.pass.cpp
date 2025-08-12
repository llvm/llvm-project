//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: FROZEN-CXX03-HEADERS-FIXME

// This test requires variable templates
// UNSUPPORTED: gcc && c++11

// <functional>

// reference_wrapper

// Ensure that std::reference_wrapper does not inhibit optimizations based on the
// std::__desugars_to internal helper.

#include <functional>
#include <__type_traits/desugars_to.h>

struct Operation {};
struct Tag {};

namespace std {
template <>
bool const __desugars_to_v<Tag, Operation> = true;
}

static_assert(std::__desugars_to_v<Tag, Operation>, "something is wrong with the test");

// make sure we pass through reference_wrapper
static_assert(std::__desugars_to_v<Tag, std::reference_wrapper<Operation> >, "");
static_assert(std::__desugars_to_v<Tag, std::reference_wrapper<Operation const> >, "");
