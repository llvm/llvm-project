//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: *

// Make sure that the test DOES NOT pass if it fails at link-time

extern void this_is_an_undefined_symbol();

int main(int, char**) {
    this_is_an_undefined_symbol();
    return 1;
}
