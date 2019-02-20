//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T, class charT, class traits>
//   basic_ostream<charT, traits>&
//   operator<<(basic_ostream<charT, traits>& o, const complex<T>& x);

#include <complex>
#include <sstream>
#include <cassert>

int main(int, char**)
{
    std::complex<double> c(1, 2);
    std::ostringstream os;
    os << c;
    assert(os.str() == "(1,2)");

  return 0;
}
