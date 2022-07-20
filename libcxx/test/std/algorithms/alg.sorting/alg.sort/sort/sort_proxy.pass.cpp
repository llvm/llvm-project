//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

#include <algorithm>
#include <cassert>
#include <vector>

void test() {
  // TODO: use a custom proxy iterator instead of (or in addition to) `vector<bool>`.
  std::vector<bool> v(5, false);
  v[1] = true; v[3] = true;
  std::sort(v.begin(), v.end());
  assert(std::is_sorted(v.begin(), v.end()));
}

int main(int, char**) {
  test();

  return 0;
}
