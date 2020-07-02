//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure the test passes if we don't have clang-verify support and
// the test fails to compile.

// UNSUPPORTED: verify-support

struct Foo { };
typedef Foo::x x;
