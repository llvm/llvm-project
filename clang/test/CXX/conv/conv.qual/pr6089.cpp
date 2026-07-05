// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

bool is_char_ptr( const char* );

template< class T >
        long is_char_ptr( T /* r */ );

// Note: the const here does not lead to a qualification conversion
template< class T >
        void    make_range( T* const r, bool );

template< class T >
        void make_range( T& r, long );

void first_finder( const char*& Search )
{
        make_range( Search, is_char_ptr(Search) );
}
