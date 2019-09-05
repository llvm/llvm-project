//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// RUN: %build -fno-exceptions
// RUN: %run

// RUN: %build
// RUN: %run

// UNSUPPORTED: c++98, c++03

// <vector>

// Test that vector won't try to call the move constructor when resizing if
// the class has a deleted move constructor (but a working copy constructor).

#include <vector>

class CopyOnly {
public:
  CopyOnly() { }

  CopyOnly(CopyOnly&&) = delete;
  CopyOnly& operator=(CopyOnly&&) = delete;

  CopyOnly(const CopyOnly&) = default;
  CopyOnly& operator=(const CopyOnly&) = default;
};

int main() {
    std::vector<CopyOnly> x;
    x.emplace_back();

    CopyOnly c;
    x.push_back(c);

    return 0;
}
