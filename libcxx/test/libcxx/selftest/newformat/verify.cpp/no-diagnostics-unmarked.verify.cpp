//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

// XFAIL: *

// Make sure the test DOES NOT pass if there are no diagnostics, but we didn't
// use the 'expected-no-diagnostics' markup.
//
// Note: For the purpose of this test, make sure the file would otherwise
//       compile to make sure we really fail due to a lack of markup.

int main() { }
