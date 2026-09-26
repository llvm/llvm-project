//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

/*
  (19.6.4.3) Observers [stacktrace.basic.obs]

  size_type max_size() const noexcept;
*/

#include <cassert>
#include <stacktrace>
#include <vector>

int main(int, char**) {
  std::stacktrace st;
  static_assert(noexcept(st.max_size()));
  assert(st.max_size() == (std::vector<std::stacktrace_entry>().max_size()));

  return 0;
}
