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
// The paper recommends `namespace cpp = std::cpp29;`. However, a lot of code uses using-directives instead.
// With the obvious implementation of release namespaces (a using-directive for std plus declarations for
// revised components), a using-directive for a release namespace makes every revised name ambiguous, because
// using-directives are transitive for unqualified lookup: both std::cpp29::vector and std::vector are found.

#include <string>
#include <vector>

void using_directive() {
  using namespace std::cpp29;

  string s;     // fine, only std::string is found
  vector<int> v; // expected-error {{reference to 'vector' is ambiguous}}
}

void using_declaration() {
  // A using-declaration works.
  using std::cpp29::vector;
  vector<int> v;
  (void)v;
}
