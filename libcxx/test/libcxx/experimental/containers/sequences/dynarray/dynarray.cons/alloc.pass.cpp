//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// dynarray.cons

// template <class Alloc>
//   dynarray(size_type c, const Alloc& alloc);
// template <class Alloc>
//   dynarray(size_type c, const T& v, const Alloc& alloc);
// template <class Alloc>
//   dynarray(const dynarray& d, const Alloc& alloc);
// template <class Alloc>
//   dynarray(initializer_list<T>, const Alloc& alloc);

// ~dynarray();


#include <__config>

#include <experimental/dynarray>
#include <cassert>

#include <algorithm>
#include <complex>
#include <string>
#include "test_allocator.h"

using std::experimental::dynarray;

template <class T, class Allocator>
void check_allocator ( const dynarray<T> &dyn, const Allocator &alloc ) {
    for ( int i = 0; i < dyn.size (); ++i )
        assert ( dyn[i].get_allocator() == alloc );
}

template <class T, class Allocator>
void test ( const std::initializer_list<T> &vals, const Allocator &alloc ) {
    typedef dynarray<T> dynA;

    dynA d1 ( vals, alloc );
    assert ( d1.size () == vals.size() );
    assert ( std::equal ( vals.begin (), vals.end (), d1.begin (), d1.end ()));
    check_allocator ( d1, alloc );
    }


template <class T, class Allocator>
void test ( const T &val, const Allocator &alloc1, const Allocator &alloc2 ) {
    typedef dynarray<T> dynA;

    dynA d1 ( 4, alloc1 );
    assert ( d1.size () == 4 );
    assert ( std::all_of ( d1.begin (), d1.end (), []( const T &item ){ return item == T(); } ));
    check_allocator ( d1, alloc1 );

    dynA d2 ( 7, val, alloc1 );
    assert ( d2.size () == 7 );
    assert ( std::all_of ( d2.begin (), d2.end (), [&val]( const T &item ){ return item == val; } ));
    check_allocator ( d2, alloc1 );

    dynA d3 ( d2, alloc2 );
    assert ( d3.size () == 7 );
    assert ( std::all_of ( d3.begin (), d3.end (), [&val]( const T &item ){ return item == val; } ));
    check_allocator ( d3, alloc2 );
    }

int main()
{
//  This test is waiting on the resolution of LWG issue #2235
//     typedef test_allocator<char> Alloc;
//     typedef std::basic_string<char, std::char_traits<char>, Alloc> nstr;
//
//     test ( nstr("fourteen"), Alloc(3), Alloc(4) );
//     test ( { nstr("1"), nstr("1"), nstr("2"), nstr("3"), nstr("5"), nstr("8")}, Alloc(6));
}

