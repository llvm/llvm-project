// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template <typename T>
struct X {};
auto b = []() {
  struct S {
    static typename X<decltype(int)>::type Run(){}; // expected-error {{expected '('}}
  };
  return 5;
}();

template <typename T1, typename T2>
class PC {
};

template <typename T>
class P {
  static typename PC<T, Invalid>::Type Foo(); // expected-error {{undeclared identifier 'Invalid'}}
};
