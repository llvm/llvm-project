//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// XFAIL: availability

// dynarray.data

// void fill(const T& v);
// const T* data() const noexcept;


#include <__config>

#include <experimental/dynarray>
#include <cassert>

#include <algorithm>
#include <complex>
#include <string>

using std::experimental::dynarray;

template <class T>
void test ( const T &val ) {
    typedef dynarray<T> dynA;

    dynA d1 ( 4 );
    d1.fill ( val );
    assert ( std::all_of ( d1.begin (), d1.end (),
                    [&val]( const T &item ){ return item == val; } ));
    }

int main()
{
    test<int> ( 14 );
    test<double> ( 14.0 );
    test<std::complex<double>> ( std::complex<double> ( 14, 0 ));
    test<std::string> ( "fourteen" );
}

