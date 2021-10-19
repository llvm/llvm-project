//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

// Make sure the test passes if it fails at compile-time with the expected
// diagnostic.

struct Foo { };
typedef Foo::x x; // expected-error{{no type named 'x' in 'Foo'}}

int main(int, char**) { return 0; }
