//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// streamsize and streamoff interconvert

#include <ios>
#include <cassert>

int main(int, char**)
{
    std::streamoff o(5);
    std::streamsize sz(o);
    assert(sz == 5);
    std::streamoff o2(sz);
    assert(o == o2);

  return 0;
}
