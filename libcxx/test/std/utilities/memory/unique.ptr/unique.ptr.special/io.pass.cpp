//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
//  Because we don't have a functioning decltype in C++03

// <memory>

// unique_ptr

// template<class CharT, class Traits, class Y, class D>
//   basic_ostream<CharT, Traits>&
//   operator<<(basic_ostream<CharT, Traits>& os, const unique_ptr<Y, D>& p);

#include <memory>
#include <sstream>
#include <cassert>

int main()
{
    std::unique_ptr<int> p(new int(3));
    std::ostringstream os;
    assert(os.str().empty());
    os << p;
    assert(!os.str().empty());
}
