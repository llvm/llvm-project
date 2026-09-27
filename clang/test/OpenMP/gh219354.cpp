// RUN: %clang_cc1 -fopenmp -fsyntax-only -verify %s

template <class T, int N> &T::foo() { // expected-error {{nested name specifier 'T' for declaration does not refer into a class, class template or class template partial specialization}}
#pragma omp simd collapse(N)
  ;
}

struct S {
  template <int N> void bar();
};

namespace NS {
template <int N> void S::bar() { // expected-error {{cannot define or redeclare 'bar' here because namespace 'NS' does not enclose namespace 'S'}}
#pragma omp simd collapse(N)
  ;
}
}

template <int M> class D;
template <int M> template <int N> void D<M>::baz() { // expected-error {{out-of-line definition of 'baz' from class 'D<M>' without definition}}
#pragma omp simd collapse(N)
  ;
}

template <int> struct E {
  void spec();
};
template <int N> void E<0>::spec() { // expected-error {{template parameter list matching the non-templated nested type 'E<0>' should be empty ('template<>')}}
#pragma omp simd collapse(N)
  ;
}

template <int> struct G {
  void fr();
};
struct H {
  template <int N> friend void G<0>::fr() { // expected-error {{template parameter list matching the non-templated nested type 'G<0>' should be empty ('template<>')}}
#pragma omp simd collapse(N)
    ;
  }
};

#pragma omp declare simd // expected-error {{function declaration is expected after 'declare simd' directive}}
template <class T, int N> void T::qux() { // expected-error {{nested name specifier 'T' for declaration does not refer into a class, class template or class template partial specialization}}
#pragma omp simd collapse(N)
  ;
}
