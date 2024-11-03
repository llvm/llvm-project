//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -fno-exceptions -D_LIBCPP_ENABLE_ASSERTIONS

#include <utility>

int main(int, char**) {
  auto guard = std::__make_exception_guard([] {});
  auto guard2 = std::move(guard);
  guard2.__complete();

  return 0;
}
