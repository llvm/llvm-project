//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// UNSUPPORTED: c++98, c++03, c++11

// dynarray.cons

// explicit dynarray(size_type c);
// dynarray(size_type c, const T& v);
// dynarray(initializer_list<T>);
// dynarray(const dynarray& d);

// ~dynarray();


#include <experimental/dynarray>
#include <cassert>

#include <algorithm>
#include <complex>
#include <limits>
#include <new>
#include <string>


using std::experimental::dynarray;

template <class T>
void testInitList( const std::initializer_list<T> &vals ) {
    typedef dynarray<T> dynA;

    dynA d1 ( vals );
    assert ( d1.size () == vals.size() );
    assert ( std::equal ( vals.begin (), vals.end (), d1.begin (), d1.end ()));
    }


template <class T>
void test ( const T &val, bool DefaultValueIsIndeterminate = false) {
    typedef dynarray<T> dynA;

    dynA d1 ( 4 );
    assert ( d1.size () == 4 );
    if (!DefaultValueIsIndeterminate) {
        assert ( std::all_of ( d1.begin (), d1.end (), []( const T &item ){ return item == T(); } ));
    }

    dynA d2 ( 7, val );
    assert ( d2.size () == 7 );
    assert ( std::all_of ( d2.begin (), d2.end (), [&val]( const T &item ){ return item == val; } ));

    dynA d3 ( d2 );
    assert ( d3.size () == 7 );
    assert ( std::all_of ( d3.begin (), d3.end (), [&val]( const T &item ){ return item == val; } ));
    }

void test_bad_length () {
    try { dynarray<int> ( std::numeric_limits<size_t>::max() / sizeof ( int ) + 1 ); }
    catch ( std::bad_array_length & ) { return ; }
    catch (...) { assert(false); }
    assert ( false );
}


int main()
{
    test<int> ( 14, /* DefaultValueIsIndeterminate */ true );       // ints don't get default initialized
    test<long> ( 0, true);
    test<double> ( 14.0, true );
    test<std::complex<double>> ( std::complex<double> ( 14, 0 ));
    test<std::string> ( "fourteen" );

    testInitList( { 1, 1, 2, 3, 5, 8 } );
    testInitList( { 1., 1., 2., 3., 5., 8. } );
    testInitList( { std::string("1"), std::string("1"), std::string("2"), std::string("3"),
                  std::string("5"), std::string("8")} );

//  Make sure we don't pick up the Allocator version here
    dynarray<long> d1 ( 20, 3 );
    assert ( d1.size() == 20 );
    assert ( std::all_of ( d1.begin (), d1.end (), []( long item ){ return item == 3L; } ));

    test_bad_length ();
}
