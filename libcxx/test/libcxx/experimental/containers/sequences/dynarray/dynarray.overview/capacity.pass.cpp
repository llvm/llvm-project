//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// dynarray.overview

// size_type size()     const noexcept;
// size_type max_size() const noexcept;
// bool      empty()    const noexcept;  

#include <__config>

#include <experimental/dynarray>
#include <cassert>

#include <algorithm>
#include <complex>
#include <string>

using std::experimental::dynarray;

template <class T>
void dyn_test ( const dynarray<T> &dyn, size_t sz ) {
    assert ( dyn.size ()     == sz );
    assert ( dyn.max_size () == sz );
    assert ( dyn.empty () == ( sz == 0 ));
    }

template <class T>
void test ( std::initializer_list<T> vals ) {
    typedef dynarray<T> dynA;
    
    dynA d1 ( vals );
    dyn_test ( d1, vals.size ());
    }

int main()
{
    test ( { 1, 1, 2, 3, 5, 8 } );
    test ( { 1., 1., 2., 3., 5., 8. } );
    test ( { std::string("1"), std::string("1"), std::string("2"), std::string("3"), 
                std::string("5"), std::string("8")} );

    test<int> ( {} );
    test<std::complex<double>> ( {} );
    test<std::string> ( {} );
}

