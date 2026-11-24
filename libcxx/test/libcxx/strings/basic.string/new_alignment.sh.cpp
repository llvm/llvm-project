//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that std::string bases its alignment on __STDCPP_DEFAULT_NEW_ALIGNMENT__

// UNSUPPORTED: c++03

// RUN: %{build} -faligned-new=32 -DALIGN=32
// RUN: %{run}
// RUN: %{build} -faligned-new=16 -DALIGN=16
// RUN: %{run}
// RUN: %{build} -faligned-new=8 -DALIGN=8
// RUN: %{run}

#include <cassert>
#include <string>

int main(int, char**) {
  std::string str;
  str.reserve(25);

  assert(str.capacity() % ALIGN == ALIGN - 1);

  return 0;
}
