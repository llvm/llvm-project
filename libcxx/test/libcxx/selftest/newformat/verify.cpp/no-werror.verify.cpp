//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure clang-verify tests distinguish warnings from errors.

// ADDITIONAL_COMPILE_FLAGS: -Wunused-variable

int main() {
    int foo; // expected-warning {{unused variable}}
}
