//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that `vector<const T>` works

#include <vector>

void test() {
  std::vector<const int> v;
  v.emplace_back(1);
  v.push_back(1);
  v.resize(3);
}
