//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// consteval text_encoding text_encoding::literal() noexcept;

#include <text_encoding>

constexpr decltype(auto) foo() {
  // expected-error@*:* {{cannot take address of consteval function 'literal' outside of an immediate invocation}}
  return &std::text_encoding::literal;
}
