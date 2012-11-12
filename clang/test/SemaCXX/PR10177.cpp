// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template<typename T, typename U, U> using alias_ref = T;
template<typename T, typename U, U> void func_ref() {}
template<typename T, typename U, U> struct class_ref {};

template<int N>
struct U {
  static int a;
};

template<int N> struct S; // expected-note 2{{here}}

template<int N>
int U<N>::a = S<N>::kError; // expected-error 2{{undefined}}

template<typename T>
void f() {
  // FIXME: The standard suggests that U<0>::a is odr-used by this expression,
  // but it's not entirely clear that's the right behaviour.
  (void)alias_ref<int, int&, U<0>::a>();
  (void)func_ref<int, int&, U<1>::a>(); // expected-note {{here}}
  (void)class_ref<int, int&, U<2>::a>(); // expected-note {{here}}
};

int main() {
  f<int>(); // expected-note 2{{here}}
}

namespace N {
  template<typename T> struct S { static int n; };
  template<typename T> int S<T>::n = 5;
  void g(int*);
  template<typename T> int f() {
    int k[S<T>::n];
    g(k);
    return k[3];
  }
  int j = f<int>();
}
