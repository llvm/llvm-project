// RUN: %clang_cc1 -std=c++2a -verify %s

namespace PR44761 {
  template<typename T> concept X = (sizeof(T) == sizeof(T));

  template<typename T> struct A {
    bool operator<(const A&) const & requires X<T>; // #1
    int operator<=>(const A&) const & requires X<T> && X<int> = delete; // #2
  };
  bool k1 = A<int>() < A<int>(); // prefer more-constrained 'operator<=>'
  // expected-error@-1 {{deleted}}
  // expected-note@#1 {{candidate}}
  // expected-note@#2 {{candidate function has been explicitly deleted}}
  // expected-note@#2 {{candidate function (with reversed parameter order) has been explicitly deleted}}
  bool k2 = A<float>() < A<float>(); // prefer more-constrained 'operator<=>'
  // expected-error@-1 {{deleted}}
  // expected-note@#1 {{candidate}}
  // expected-note@#2 {{candidate function has been explicitly deleted}}
  // expected-note@#2 {{candidate function (with reversed parameter order) has been explicitly deleted}}
}

namespace ambiguous_resolution {
  template<class T> struct S {
    int f() const requires true { return 1; } // #S-const-overload
    int f() volatile { return 2; } // #S-volatile-overload
  };
  int test() {
    // Here, we have two overloads: `const S&` and `volatile S&`
    // Neither conversion should win the tie-break, and so we should
    // instead error on ambiguous overloads
    S<int> s; return s.f();
    // expected-error@-1 {{call to member function 'f' is ambiguous}}
    // expected-note@#S-const-overload {{candidate function}}
    // expected-note@#S-volatile-overload {{candidate function}}
  }
}
