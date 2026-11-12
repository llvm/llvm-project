// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s

template<typename T>
struct X {
    using Y = typename T::invalid;
};

template<typename T>
concept Invalid = X<T>{};

template<typename T>
concept False = false; // expected-note{{because 'false' evaluated to false}}

template<typename T>
concept True = true;

// TODO: Concepts: Uncomment trailing requires clauses here when we have correct substitution.
//template<True T>
//  requires False<T>
//void g1() requires Invalid<T>;
//
//using g1i = decltype(g1<int>());

template<False T> // expected-note{{because 'int' does not satisfy 'False'}}
  requires Invalid<T>
void g2(); // requires Invalid<T>;
// expected-note@-1{{candidate template ignored: constraints not satisfied [with T = int]}}

using g2i = decltype(g2<int>());
// expected-error@-1{{no matching function for call to 'g2'}}