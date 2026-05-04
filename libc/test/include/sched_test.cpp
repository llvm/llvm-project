//===-- Unittests for sched -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sched.h>

template <typename T, typename U> struct SameType {
  static constexpr bool value = false;
};

template <typename T> struct SameType<T, T> {
  static constexpr bool value = true;
};

// Use unevaluated contexts to verify the public macro declarations without
// requiring this include test to link the helper entrypoints.
static_assert(SameType<decltype(CPU_ZERO((cpu_set_t *)0)), void>::value, "");
static_assert(SameType<decltype(CPU_COUNT((cpu_set_t *)0)), int>::value, "");
static_assert(SameType<decltype(CPU_SET(0, (cpu_set_t *)0)), void>::value, "");
static_assert(SameType<decltype(CPU_ISSET(0, (cpu_set_t *)0)), int>::value, "");

extern "C" int main() { return 0; }
