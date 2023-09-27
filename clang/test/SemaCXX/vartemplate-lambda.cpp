// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s

template <class> auto fn0 = [] {};
template <typename> void foo0() { fn0<char>(); }

template<typename T> auto fn1 = [](auto a) { return a + T(1); };
template<typename T> auto v1 = [](int a = T()) { return a; }();
// expected-error@-1{{cannot initialize a parameter of type 'int' with an rvalue of type 'int *'}}
// expected-note@-2{{in instantiation of default function argument expression for 'operator()<int *>' required here}}
// expected-note@-3{{passing argument to parameter 'a' here}}
// expected-note@-4{{substituting into a lambda}}

struct S {
  template<class T>
  static constexpr T t = [](int f = T(7)){return f;}(); // expected-error{{a lambda expression may not appear inside of a constant expression}}
};

template <typename X>
int foo2() {
  X a = 0x61;
  fn1<char>(a);
  (void)v1<int>;
  (void)v1<int *>; // expected-note{{in instantiation of variable template specialization 'v1' requested here}}
  (void)S::t<int>;
  return 0;
}

template<class C>
int foo3() {
  C::m1(); // expected-error{{type 'long long' cannot be used prior to '::' because it has no members}}
  return 1;
}

template<class C>
auto v2 = [](int a = foo3<C>()){};  // expected-note{{in instantiation of function template specialization 'foo3<long long>' requested here}}

int main() {
  v2<long long>();  // This line causes foo3<long long> to be instantiated.
  v2<long long>(2); // This line does not.
  foo2<int>();
}
