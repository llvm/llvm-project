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

// dynarray.overview

// reference       front();
// const_reference front() const;
// reference       back();
// const_reference back()  const;


#include <experimental/dynarray>
#include <cassert>

#include <algorithm>
#include <complex>
#include <string>

using std::experimental::dynarray;

template <class T>
void dyn_test_const ( const dynarray<T> &dyn, bool CheckValues = true ) {
    const T *data = dyn.data ();
    assert(data == &dyn.front());
    assert((data + dyn.size() - 1) == &dyn.back());
    if (CheckValues) {
        assert ( *data == dyn.front ());
        assert ( *(data + dyn.size() - 1 ) == dyn.back ());
    }
}

template <class T>
void dyn_test ( dynarray<T> &dyn, bool CheckValues = true ) {
    T *data = dyn.data ();
    assert(data == &dyn.front());
    assert((data + dyn.size() - 1) == &dyn.back());
    if (CheckValues) {
        assert ( *data == dyn.front ());
        assert ( *(data + dyn.size() - 1 ) == dyn.back ());
    }
}


template <class T>
void test ( const T &val, bool DefaultValueIsIndeterminate = false) {
    typedef dynarray<T> dynA;

    const bool CheckDefaultValues = ! DefaultValueIsIndeterminate;

    dynA d1 ( 4 );
    dyn_test ( d1, CheckDefaultValues );
    dyn_test_const ( d1, CheckDefaultValues );

    dynA d2 ( 7, val );
    dyn_test ( d2 );
    dyn_test_const ( d2 );
}

int main()
{
    test<int> ( 14, /* DefaultValueIsIndeterminate */ true);
    test<double> ( 14.0, true );
    test<std::complex<double>> ( std::complex<double> ( 14, 0 ));
    test<std::string> ( "fourteen" );
}
