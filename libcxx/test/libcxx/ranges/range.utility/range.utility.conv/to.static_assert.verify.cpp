//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated-volatile

// Test that the "mandates" requirements on the given container are checked using `static_assert`.

#include <ranges>
#include <vector>

void test() {
  using R = std::vector<int>;
  R in = {1, 2, 3};

  (void)std::ranges::to<const R>(in); //expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}The target container cannot be const-qualified, please remove the const}}
  (void)(in | std::ranges::to<const R>()); //expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}The target container cannot be const-qualified, please remove the const}}
  (void)std::ranges::to<volatile R>(in); //expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}The target container cannot be volatile-qualified, please remove the volatile}}
  (void)(in | std::ranges::to<volatile R>()); //expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}The target container cannot be volatile-qualified, please remove the volatile}}
}
