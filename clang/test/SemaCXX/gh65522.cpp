// RUN: %clang_cc1 -std=c++20 -Wc++17-compat -verify -Wno-unused %s

class X {};

template<typename T>
class B3 { // expected-note {{candidate template ignored: could not match 'B3<T>' against 'int'}} \
           // expected-note {{implicit deduction guide declared as 'template <typename T> B3(B3<T>) -> B3<T>'}}
  template<X x> B3(T); // expected-warning 2{{non-type template parameter of type 'X' is incompatible with C++ standards before C++20}} \
                       // expected-note {{candidate template ignored: couldn't infer template argument 'x'}} \
                       // expected-note {{implicit deduction guide declared as 'template <typename T, X x> B3(T) -> B3<T>'}}
};
B3 b3 = 0; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'B3'}} \
           // expected-note {{while building implicit deduction guide first needed here}}
