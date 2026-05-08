// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template <int> struct bad {
  template <class T, auto =
                         [] { // #lambda
                           // expected-note@#lambda {{while substituting into a lambda expression here}}
                           // expected-note@#lambda 2{{capture 'i' by value}}
                           // expected-note@#lambda 2{{capture 'i' by reference}}
                           // expected-note@#lambda 2{{default capture by value}}
                           // expected-note@#lambda 2{{default capture by reference}}
                           for (int i = 0; i < 100; ++i) { // #i
                             // expected-error@-1 {{variable 'i' cannot be implicitly captured in a lambda with no capture-default specified}}
                             // expected-note@#i {{'i' declared here}}
                             // expected-note@#lambda {{lambda expression begins here}}
                             // expected-error@-4 {{variable 'i' cannot be implicitly captured in a lambda with no capture-default specified}}
                             // expected-note@#i {{'i' declared here}}
                             // expected-note@#lambda {{lambda expression begins here}}
                             struct LoopHelper {
                               static constexpr void process() {}
                             };
                           }
                         }>
  static void f(T) {} // expected-note {{in instantiation of default argument for 'f<int>' required here}}
};

int main() { bad<0>::f(0); } // expected-note {{while substituting deduced template arguments into function template 'f'}}
