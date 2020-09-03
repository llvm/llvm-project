//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure the test passes even if there's a link error, i.e. it isn't linked.

extern void this_is_an_undefined_symbol();

int main() {
    this_is_an_undefined_symbol();
}
