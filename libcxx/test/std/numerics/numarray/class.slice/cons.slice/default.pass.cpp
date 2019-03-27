//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// class slice;

// slice();

#include <valarray>
#include <cassert>

int main(int, char**)
{
    std::slice s;
    assert(s.start() == 0);
    assert(s.size() == 0);
    assert(s.stride() == 0);

  return 0;
}
