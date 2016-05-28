//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

#include <tuple>
#include <string>
#include <complex>

#include <cassert>

int main()
{
    typedef std::complex<float> cf;
    auto t1 = std::make_tuple<double, int, std::string, cf, int> ( 42, 21, "Hi", { 1,2 } );
    assert ( std::get<int>(t1) == 42 ); // two ints here (one at the end)
}
