//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://github.com/llvm/llvm-project/issues/40340 is fixed
// UNSUPPORTED: availability-pmr-missing

// <unordered_set>

// namespace std::pmr {
//
// typedef ... unordered_set
//
// } // namespace std::pmr

#include <unordered_set>

int main(int, char**) {
  {
    // Check that std::pmr::unordered_set is usable without <memory_resource>.
    std::pmr::unordered_set<int> s;
    std::pmr::unordered_multiset<int> ms;
  }

  return 0;
}
