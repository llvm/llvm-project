// The intention of this file to check we could only export declarations in namesapce scope.
//
// RUN: %clang_cc1 -std=c++20 %s -verify

export module X;

export template <typename T>
struct X {
  struct iterator {
    T node;
  };
  void foo() {}
  template <typename U>
  U bar();
};

export template <typename T> struct X<T>::iterator;               // expected-error {{cannot export 'iterator' as it is not at namespace scope}}
export template <typename T> void X<T>::foo();                    // expected-error {{cannot export 'foo' as it is not at namespace scope}}
export template <typename T> template <typename U> U X<T>::bar(); // expected-error {{cannot export 'bar' as it is not at namespace scope}}

export struct Y {
  struct iterator {
    int node;
  };
  void foo() {}
  template <typename U>
  U bar();
};

export struct Y::iterator;               // expected-error {{cannot export 'iterator' as it is not at namespace scope}}
export void Y::foo();                    // expected-error {{cannot export 'foo' as it is not at namespace scope}}
export template <typename U> U Y::bar(); // expected-error {{cannot export 'bar' as it is not at namespace scope}}

export {
  template <typename T> struct X<T>::iterator; // expected-error {{cannot export 'iterator' as it is not at namespace scope}}
  struct Y::iterator;                          // expected-error {{cannot export 'iterator' as it is not at namespace scope}}
}
