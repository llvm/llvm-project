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

// const_reference at(size_type n) const;
//       reference at(size_type n);
  
#include <__config>

#include <experimental/dynarray>
#include <cassert>

#include <algorithm>
#include <complex>
#include <string>

using std::experimental::dynarray;

template <class T>
void dyn_test_const ( const dynarray<T> &dyn, const std::initializer_list<T> &vals ) {
    const T *data = dyn.data ();
    auto it = vals.begin ();
    for ( size_t i = 0; i < dyn.size(); ++i, ++it ) {
        assert ( data + i == &dyn[i]);
        assert ( *it == dyn[i]);
        }
    }

template <class T>
void dyn_test ( dynarray<T> &dyn, const std::initializer_list<T> &vals ) {
    T *data = dyn.data ();
    auto it = vals.begin ();
    for ( size_t i = 0; i < dyn.size(); ++i, ++it ) {
        assert ( data + i == &dyn[i]);
        assert ( *it == dyn[i]);
        }
    }


template <class T>
void test ( std::initializer_list<T> vals ) {
    typedef dynarray<T> dynA;
    
    dynA d1 ( vals );
    dyn_test ( d1, vals );
    dyn_test_const ( d1, vals );
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

