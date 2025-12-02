//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

// <print>

// The Standard does indirectly require that <print> includes <format>. However
// using the granularized headers so it's possible to implement <print> without
// <format>. This would be a non-conforming implementation.
//
// See https://llvm.org/PR71925

#include <print>
#include <vector>

void test() {
  std::vector<int> v{1, 2, 3};
  std::print("{} {}", 42, v);
}
