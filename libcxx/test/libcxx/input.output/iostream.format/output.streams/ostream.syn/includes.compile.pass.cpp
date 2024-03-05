//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

// <ostream>

// The Standard does indirectly require that <ostream> includes <format>.
// However using the granularized headers so it's possible to implement
// <ostream> without <format>. This would be a non-conforming implementation.
//
// See https://github.com/llvm/llvm-project/issues/71925

#include <ostream>
#include <vector>

extern std::ostream& os;

void test() {
  std::vector<int> v{1, 2, 3};
  std::print(os, "{} {}", 42, v);
}
