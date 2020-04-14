//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: objective-c++

// Make sure the test passes if it fails at compile-time, with verify

struct Foo { };
typedef Foo::x x; // expected-error{{no type named 'x' in 'Foo'}}

int main() { }
