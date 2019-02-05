//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// bool operator!() const;

#include <ios>
#include <cassert>

int main(int, char**)
{
    std::ios ios(0);
    assert(!ios == ios.fail());
    ios.setstate(std::ios::failbit);
    assert(!ios == ios.fail());

  return 0;
}
