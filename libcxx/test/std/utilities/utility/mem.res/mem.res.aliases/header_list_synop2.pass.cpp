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

// <list>

// namespace std::pmr {
//
// typedef ... list
//
// } // namespace std::pmr

#include <list>

int main(int, char**) {
  {
    // Check that std::pmr::list is usable without <memory_resource>.
    std::pmr::list<int> l;
  }

  return 0;
}
