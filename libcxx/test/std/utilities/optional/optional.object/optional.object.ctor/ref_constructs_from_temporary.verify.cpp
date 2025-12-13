//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// optional

#include <optional>
#include <utility>

struct X {
  int i;

  X(int j) : i(j) {}
};

int main(int, char**) {
  const std::optional<int> _co(1);
  std::optional<int> _o(1);

  // expected-error-re@*:* 8 {{call to deleted constructor of 'std::optional<{{.*}}>'}}
  std::optional<const int&> o1{1};                     // optional(U&&)
  std::optional<const int&> o2{std::optional<int>(1)}; // optional(optional<U>&&)
  std::optional<const int&> o3{_co};                   // optional(const optional<U>&)
  std::optional<const int&> o4{_o};                    // optional(optional<U>&)
  std::optional<const X&> o5{1};                       // optional(U&&)
  std::optional<const X&> o6{std::optional<int>(1)};   // optional(optional<U>&&)
  std::optional<const X&> o7{_co};                     // optional(const optional<U>&)
  std::optional<const X&> o8{_o};                      // optional(optional<U>&)
}
