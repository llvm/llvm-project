//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// PROTOTYPE: P3866R0 (V2: An Evolution Path for the Standard Library)
//
// std::vector and std::vector2 are unrelated types. Code that moves to `cpp::vector` can't pass its vectors to
// existing APIs that take a std::vector, and vice versa.

#include <vector>

namespace cpp = std::cpp29;

void legacy_api(std::vector<int> const&); // expected-note {{candidate function not viable}}
void modern_api(cpp::vector<int> const&); // expected-note {{candidate function not viable}}

void f() {
  cpp::vector<int> v2;
  std::vector<int> v1;

  legacy_api(v2); // expected-error {{no matching function for call to 'legacy_api'}}
  modern_api(v1); // expected-error {{no matching function for call to 'modern_api'}}
}
