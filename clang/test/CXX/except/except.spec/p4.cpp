// RUN: %clang_cc1 -std=c++11 %s -verify -fcxx-exceptions

// We permit overriding an implicit exception specification with an explicit one
// as an extension, for compatibility with existing code.

struct S {
  void a(); // expected-note {{here}}
  ~S(); // expected-note {{here}}
  void operator delete(void*); // expected-note {{here}}
};

void S::a() noexcept {} // expected-error {{does not match previous}}
S::~S() noexcept {} // expected-warning {{function previously declared with an implicit exception specification redeclared with an explicit exception specification}}
void S::operator delete(void*) noexcept {} // expected-warning {{function previously declared with an implicit exception specification redeclared with an explicit exception specification}}

struct T {
  void a() noexcept; // expected-note {{here}}
  ~T() noexcept; // expected-note {{here}}
  void operator delete(void*) noexcept; // expected-note {{here}}
};

void T::a() {} // expected-warning {{missing exception specification 'noexcept'}}
T::~T() {} // expected-warning {{function previously declared with an explicit exception specification redeclared with an implicit exception specification}}
void T::operator delete(void*) {} // expected-warning {{function previously declared with an explicit exception specification redeclared with an implicit exception specification}}


// The extension does not extend to function templates.

template<typename T> struct U {
  T t;
  ~U(); // expected-note {{here}}
  void operator delete(void*); // expected-note {{here}}
};

template<typename T> U<T>::~U() noexcept(true) {} // expected-error {{exception specification in declaration does not match previous declaration}}
template<typename T> void U<T>::operator delete(void*) noexcept(false) {} // expected-error {{exception specification in declaration does not match previous declaration}}
