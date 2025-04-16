//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20


// Test that the "mandates" requirements on the given class are checked using `static_assert`.
#include <ranges>

void test() {
  struct R {
    int* begin() const { reurn nullptr; };
    int* end() const { return nullptr; };

    operator int() const { return 0; }
  };
  (void)std::ranges::to<int>(R{});
  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<int>());
  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  
}