//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: *

// Make sure the test DOES NOT pass if we don't have clang-verify support and
// the test compiles successfully.

// UNSUPPORTED: verify-support

int main() { }
