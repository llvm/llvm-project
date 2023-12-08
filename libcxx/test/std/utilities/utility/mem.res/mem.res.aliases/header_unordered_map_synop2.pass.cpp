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

// <unordered_map>

// namespace std::pmr {
//
// typedef ... unordered_map
//
// } // namespace std::pmr

#include <unordered_map>

int main(int, char**) {
  {
    // Check that std::pmr::unordered_map is usable without <memory_resource>.
    std::pmr::unordered_map<int, int> m;
    std::pmr::unordered_multimap<int, int> mm;
  }

  return 0;
}
