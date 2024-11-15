// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify=expected,cxx11 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify=expected,since-cxx14 %s

struct A {
  template<typename T>
  void f0();

  template<>
  constexpr void f0<short>(); // cxx11-error {{conflicting types for 'f0'}}
                              // cxx11-note@-1 {{previous declaration is here}}
                              // cxx11-warning@-2 {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const'}}

  template<typename T>
  void f1() const; // since-cxx14-note 2{{candidate template ignored: could not match 'void () const' against 'void ()'}}

  template<>
  constexpr void f1<short>(); // since-cxx14-error {{no function template matches function template specialization 'f1'}}
                              // cxx11-warning@-1 {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const'}}
};

template<>
constexpr void A::f0<long>(); // cxx11-error {{conflicting types for 'f0'}}
                              // cxx11-note@-1 {{previous declaration is here}}
                              // cxx11-warning@-2 {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const'}}

template<>
constexpr void A::f1<long>(); // since-cxx14-error {{no function template matches function template specialization 'f1'}}
                              // cxx11-warning@-1 {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const'}}

// FIXME: It's unclear whether [temp.expl.spec]p12 is intended to apply to
// members of a class template explicitly specialized for an implicitly
// instantiated specialization of that template.
template<typename T>
struct B { // #defined-here
  void g0(); // since-cxx14-note {{previous declaration is here}}
             // cxx11-note@-1 {{member declaration does not match because it is not const qualified}}

  void g1() const; // since-cxx14-note {{member declaration does not match because it is const qualified}}
                   // cxx11-note@-1 {{previous declaration is here}}

  template<typename U>
  void h0(); // since-cxx14-note {{previous declaration is here}}

  template<typename U>
  void h1() const; // cxx11-note {{previous declaration is here}}
};

template<>
constexpr void B<short>::g0(); // since-cxx14-error {{constexpr declaration of 'g0' follows non-constexpr declaration}}
                               // cxx11-error@-1 {{out-of-line declaration of 'g0' does not match any declaration in 'B<short>'}}
                               // cxx11-warning@-2 {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const'}}
                               // expected-note@#defined-here {{defined here}}

template<>
constexpr void B<short>::g1(); // since-cxx14-error {{out-of-line declaration of 'g1' does not match any declaration in 'B<short>'}}
                               // cxx11-error@-1 {{constexpr declaration of 'g1' follows non-constexpr declaration}}
                               // cxx11-warning@-2 {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const'}}
                               // expected-note@#defined-here {{defined here}}

template<>
template<typename U>
constexpr void B<long>::h0(); // since-cxx14-error {{constexpr declaration of 'h0' follows non-constexpr declaration}}
                              // cxx11-error@-1 {{out-of-line declaration of 'h0' does not match any declaration in 'B<long>'}}
                              // cxx11-warning@-2 {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const'}}

template<>
template<typename U>
constexpr void B<long>::h1(); // since-cxx14-error {{out-of-line declaration of 'h1' does not match any declaration in 'B<long>'}}
                              // cxx11-error@-1 {{constexpr declaration of 'h1' follows non-constexpr declaration}}
                              // cxx11-warning@-2 {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const'}}
