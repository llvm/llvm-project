//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: libcpp-has-no-global-filesystem-namespace

#include <cstdio>

int main(int, char**) {
    // fopen is not available on systems without a global filesystem namespace.
    std::fopen("", "");

  return 0;
}
