//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// This test ensures that we diagnose when an incomplete type is used in one of
// vector's methods. The Standard requires that to be the case, and we want to
// uniformly produce an error for that. Note that producing the same diagnostic
// in all cases is difficult, but we at least want to fail to fight back against
// Hyrum's law.

#include <vector>

struct Incomplete;

void f(std::vector<Incomplete>& v) {
  (void)v.empty();  // expected-error@*:* {{}}
  (void)v.size();   // expected-error@*:* {{}}
  (void)v.begin();  // expected-error@*:* {{}}
  (void)v.end();    // expected-error@*:* {{}}
  (void)v.cbegin(); // expected-error@*:* {{}}
  (void)v.cend();   // expected-error@*:* {{}}

  // etc for other APIs
}
