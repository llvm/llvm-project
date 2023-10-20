// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++1y -verify %s -DCXX1Y

#ifndef CXX1Y

template<typename T, typename U, U> using alias_ref = T;
template<typename T, typename U, U> void func_ref() {}
template<typename T, typename U, U> struct class_ref {};

template<int N>
struct U {
  static int a;
};

template<int N> struct S; // expected-note 6{{here}}

template<int N>
int U<N>::a = S<N>::kError; // expected-error 6{{undefined}}

template<typename T>
void f() {
  (void)alias_ref<int, int&, U<0>::a>(); // expected-note {{here}}
  (void)func_ref<int, int&, U<1>::a>(); // expected-note {{here}}
  (void)class_ref<int, int&, U<2>::a>(); // expected-note {{here}}
};

template<typename T>
void not_instantiated() {
  // These cases (arguably) do not require instantiation of U<i>::a.
  (void)alias_ref<int, int&, U<3>::a>();
  (void)func_ref<int, int&, U<4>::a>();
  (void)class_ref<int, int&, U<5>::a>();
};

template<int N>
void fi() {
  (void)alias_ref<int, int&, U<N>::a>(); // expected-note {{here}}
  (void)func_ref<int, int&, U<N+1>::a>(); // expected-note {{here}}
  (void)class_ref<int, int&, U<N+2>::a>(); // expected-note {{here}}
};

int main() {
  f<int>();   // expected-note 3{{here}}
  fi<10>();   // expected-note 3{{here}}
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

#else

namespace { template<typename> extern int n; }
template<typename T> int g() { return n<int>; }
namespace { extern template int n<int>; } // expected-error {{explicit instantiation declaration of 'n<int>' with internal linkage}}

#endif

