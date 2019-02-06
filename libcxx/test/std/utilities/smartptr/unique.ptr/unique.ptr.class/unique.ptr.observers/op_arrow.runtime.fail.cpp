//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op->()

#include <memory>
#include <cassert>

struct V {
  int member;
};

int main() {
  std::unique_ptr<V[]> p;
  std::unique_ptr<V[]> const& cp = p;

  p->member; // expected-error {{member reference type 'std::unique_ptr<V []>' is not a pointer}}
  // expected-error@-1 {{no member named 'member'}}

  cp->member; // expected-error {{member reference type 'const std::unique_ptr<V []>' is not a pointer}}
              // expected-error@-1 {{no member named 'member'}}
}
