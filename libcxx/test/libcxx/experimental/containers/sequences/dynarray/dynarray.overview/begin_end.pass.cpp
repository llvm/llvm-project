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


// iterator       begin()        noexcept;
// const_iterator begin()  const noexcept;
// const_iterator cbegin() const noexcept;
// iterator       end()          noexcept;
// const_iterator end()    const noexcept;
// const_iterator cend()   const noexcept;
//
// reverse_iterator       rbegin()        noexcept;
// const_reverse_iterator rbegin()  const noexcept;
// const_reverse_iterator crbegin() const noexcept;
// reverse_iterator       rend()          noexcept;
// const_reverse_iterator rend()    const noexcept;
// const_reverse_iterator crend()   const noexcept;


#include <__config>

#include <experimental/dynarray>
#include <cassert>

#include <algorithm>
#include <complex>
#include <string>

using std::experimental::dynarray;

template <class T>
void dyn_test_const ( const dynarray<T> &dyn ) {
    const T *data = dyn.data ();
    assert ( data == &*dyn.begin ());
    assert ( data == &*dyn.cbegin ());

    assert ( data + dyn.size() - 1 == &*dyn.rbegin ());
    assert ( data + dyn.size() - 1 == &*dyn.crbegin ());

    assert ( dyn.size () == std::distance ( dyn.begin(), dyn.end()));
    assert ( dyn.size () == std::distance ( dyn.cbegin(), dyn.cend()));
    assert ( dyn.size () == std::distance ( dyn.rbegin(), dyn.rend()));
    assert ( dyn.size () == std::distance ( dyn.crbegin(), dyn.crend()));

    assert (   dyn.begin ()  ==   dyn.cbegin ());
    assert ( &*dyn.begin ()  == &*dyn.cbegin ());
    assert (   dyn.rbegin () ==   dyn.crbegin ());
    assert ( &*dyn.rbegin () == &*dyn.crbegin ());
    assert (   dyn.end ()    ==   dyn.cend ());
    assert (   dyn.rend ()   ==   dyn.crend ());
    }

template <class T>
void dyn_test ( dynarray<T> &dyn ) {
    T *data = dyn.data ();
    assert ( data == &*dyn.begin ());
    assert ( data == &*dyn.cbegin ());

    assert ( data + dyn.size() - 1 == &*dyn.rbegin ());
    assert ( data + dyn.size() - 1 == &*dyn.crbegin ());

    assert ( dyn.size () == std::distance ( dyn.begin(), dyn.end()));
    assert ( dyn.size () == std::distance ( dyn.cbegin(), dyn.cend()));
    assert ( dyn.size () == std::distance ( dyn.rbegin(), dyn.rend()));
    assert ( dyn.size () == std::distance ( dyn.crbegin(), dyn.crend()));

    assert (   dyn.begin ()  ==   dyn.cbegin ());
    assert ( &*dyn.begin ()  == &*dyn.cbegin ());
    assert (   dyn.rbegin () ==   dyn.crbegin ());
    assert ( &*dyn.rbegin () == &*dyn.crbegin ());
    assert (   dyn.end ()    ==   dyn.cend ());
    assert (   dyn.rend ()   ==   dyn.crend ());
    }


template <class T>
void test ( const T &val ) {
    typedef dynarray<T> dynA;

    dynA d1 ( 4 );
    dyn_test ( d1 );
    dyn_test_const ( d1 );

    dynA d2 ( 7, val );
    dyn_test ( d2 );
    dyn_test_const ( d2 );
    }

int main()
{
    test<int> ( 14 );
    test<double> ( 14.0 );
    test<std::complex<double>> ( std::complex<double> ( 14, 0 ));
    test<std::string> ( "fourteen" );
}

