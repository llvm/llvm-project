//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// basic_ostream<charT,traits>* tie(basic_ostream<charT,traits>* tiestr);

#include <ios>
#include <cassert>

int main(int, char**)
{
    std::ios ios(0);
    std::ostream* os = (std::ostream*)1;
    std::ostream* r = ios.tie(os);
    assert(r == 0);
    assert(ios.tie() == os);

  return 0;
}
