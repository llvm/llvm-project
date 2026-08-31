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

  allocator_type get_allocator() const noexcept;
*/

#include <cassert>
#include <concepts>
#include <stacktrace>

int main(int, char**) {
  std::stacktrace const st;
  static_assert(noexcept(st.get_allocator()));
  static_assert(std::same_as<decltype(st.get_allocator()), std::stacktrace::allocator_type>);
  return 0;
}
