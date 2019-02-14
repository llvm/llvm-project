//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class StateT> class fpos

// converts to and from streamoff

#include <ios>
#include <cassert>

int main(int, char**)
{
    typedef std::fpos<std::mbstate_t> P;
    P p(std::streamoff(7));
    std::streamoff offset(p);
    assert(offset == 7);

  return 0;
}
