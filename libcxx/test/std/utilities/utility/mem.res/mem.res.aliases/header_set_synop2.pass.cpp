//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: availability-pmr-missing

// <set>

// namespace std::pmr {
//
// typedef ... set
//
// } // namespace std::pmr

#include <set>

int main(int, char**) {
  {
    // Check that std::pmr::set is usable without <memory_resource>.
    std::pmr::set<int> s;
    std::pmr::multiset<int> ms;
  }

  return 0;
}
