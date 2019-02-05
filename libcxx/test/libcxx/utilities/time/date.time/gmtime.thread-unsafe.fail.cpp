//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: libcpp-has-no-thread-unsafe-c-functions

#include <ctime>

int main() {
    // gmtime is not thread-safe.
    std::time_t t = 0;
    std::gmtime(&t);
}
