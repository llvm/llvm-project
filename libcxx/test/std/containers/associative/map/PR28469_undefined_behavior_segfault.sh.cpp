//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %build -O2
// RUN: %run

// <map>

// Previously this code caused a segfault when compiled at -O2 due to undefined
// behavior in __tree. See https://bugs.llvm.org/show_bug.cgi?id=28469

#include <functional>
#include <map>

void dummy() {}

struct F {
    std::map<int, std::function<void()> > m;
    F() { m[42] = &dummy; }
};

int main(int, char**) {
    F f;
    f = F();

  return 0;
}
