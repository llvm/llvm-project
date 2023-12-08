// RUN:  %clang_cc1 -std=c++2a -verify %s

// Make sure constraint expressions are unevaluated before being substituted
// into during satisfaction checking.

template<typename T> constexpr bool f = T::value;
// expected-error@-1 4{{type}}

namespace unevaluated {
  template<typename T> concept Foo = false && f<int>;
  bool k = Foo<int>;
  template<typename T> requires false && f<int> struct S {};
  // expected-note@-1{{because}}
  using s = S<int>; // expected-error {{constraints not satisfied}}
  template<typename T> void foo() requires false && f<int> { };
  // expected-note@-1{{because}} expected-note@-1{{candidate template ignored}}
  int a = (foo<int>(), 0); // expected-error{{no matching function}}
  template<typename T> void bar() requires requires { requires false && f<int>; } { };
  // expected-note@-1{{because}} expected-note@-1{{candidate template ignored}}
  int b = (bar<int>(), 0); // expected-error{{no matching function}}
  template<typename T> struct M { static void foo() requires false && f<int> { }; };
  // expected-note@-1{{because}}
  int c = (M<int>::foo(), 0);
  // expected-error@-1{{invalid reference to function 'foo': constraints not satisfied}}
}

namespace constant_evaluated {
  template<typename T> requires f<int[0]> struct S {};
  // expected-note@-1{{in instantiation of}} expected-note@-1{{while substituting}}
  using s = S<int>;
  // expected-note@-1 {{while checking}}
  template<typename T> void foo() requires f<int[1]> { };
  // expected-note@-1{{in instantiation}} expected-note@-1{{while substituting}} \
     expected-note@-1{{candidate template ignored}}
  int a = (foo<int>(), 0);
  // expected-note@-1 {{while checking}} expected-error@-1{{no matching function}} \
     expected-note@-1 {{in instantiation}}
  template<typename T> void bar() requires requires { requires f<int[2]>; } { };
  // expected-note@-1{{in instantiation}} \
     expected-note@-1{{while substituting}} \
     expected-note@-1 {{while checking the satisfaction of nested requirement}}
  int b = (bar<int>(), 0);
  template<typename T> struct M { static void foo() requires f<int[3]> { }; };
  // expected-note@-1{{in instantiation}} expected-note@-1{{while substituting}}
  int c = (M<int>::foo(), 0);
  // expected-note@-1 {{while checking}}
}
