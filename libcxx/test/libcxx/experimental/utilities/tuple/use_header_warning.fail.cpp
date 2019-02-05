//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

// <experimental/tuple>

#include <experimental/tuple>

// expected-error@experimental/tuple:* {{"<experimental/tuple> has been removed. Use <tuple> instead."}}

int main(int, char**) {
  return 0;
}
