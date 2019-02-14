//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
#include <utility>
#include <complex>

#include <cassert>

int main(int, char**)
{
    typedef std::complex<float> cf;
    auto t1 = std::make_pair<int, double> ( 42, 3.4 );
    assert (( std::get<cf>(t1) == cf {1,2} ));  // no such type

  return 0;
}
