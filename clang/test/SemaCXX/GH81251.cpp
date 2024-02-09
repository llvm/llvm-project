// RUN: %clang_cc1 -fsyntax-only -verify %s

template < class T, class V > struct A
{
    A ();
    A (A &);
    A (A < V,T >);
    // expected-error@-1 {{copy constructor must pass its first argument by reference}}
};

void f ()
{
    A <int, int> (A < int, int >());
    // expected-note@-1 {{in instantiation of template class 'A<int, int>' requested here}}
    
    A <int, double> (A < int, double >());
}
