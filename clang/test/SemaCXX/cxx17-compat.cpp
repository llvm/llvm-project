// RUN: %clang_cc1 -fsyntax-only -std=c++17 -pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++2a -Wc++17-compat-pedantic -verify %s

struct A {};
int (A::*pa)() const&;
int use_pa = (A().*pa)();
#if __cplusplus <= 201703L
  // expected-warning@-2 {{invoking a pointer to a 'const &' member function on an rvalue is a C++2a extension}}
#else
  // expected-warning@-4 {{invoking a pointer to a 'const &' member function on an rvalue is incompatible with C++ standards before C++2a}}
#endif

struct B {
  void b() {
    (void) [=, this] {};
#if __cplusplus <= 201703L
    // expected-warning@-2 {{explicit capture of 'this' with a capture default of '=' is a C++2a extension}}
#else
    // expected-warning@-4 {{explicit capture of 'this' with a capture default of '=' is incompatible with C++ standards before C++2a}}
#endif
  }

  int n : 5 = 0;
#if __cplusplus <= 201703L
    // expected-warning@-2 {{default member initializer for bit-field is a C++2a extension}}
#else
    // expected-warning@-4 {{default member initializer for bit-field is incompatible with C++ standards before C++2a}}
#endif
};
