//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// Ensure that std::reference_wrapper does not inhibit optimizations based on the
// std::__desugars_to internal helper.

#include <functional>

static_assert(std::__desugars_to_v<std::__equal_tag, std::equal_to<void>, int, int>,
              "something is wrong with the test");

// make sure we pass through reference_wrapper
static_assert(std::__desugars_to_v<std::__equal_tag, std::reference_wrapper<std::equal_to<void> >, int, int>, "");
static_assert(std::__desugars_to_v<std::__equal_tag, std::reference_wrapper<std::equal_to<void> const>, int, int>, "");
