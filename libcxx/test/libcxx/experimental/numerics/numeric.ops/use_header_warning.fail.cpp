//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

// <experimental/numeric>

#include <experimental/numeric>

// expected-error@experimental/numeric:* {{"<experimental/numeric> has been removed. Use <numeric> instead."}}

int main(int, char**) {
  return 0;
}
