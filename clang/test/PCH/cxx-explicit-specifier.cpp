// RUN: %clang_cc1 -std=c++2a -include %s %s -ast-print -verify | FileCheck %s
//
// RUN: %clang_cc1 -std=c++2a -emit-pch %s -o %t-cxx2a
// RUN: %clang_cc1 -std=c++2a -include-pch %t-cxx2a %s -ast-print -verify | FileCheck %s

// RUN: %clang_cc1 -std=c++2a -emit-pch -fpch-instantiate-templates %s -o %t-cxx2a
// RUN: %clang_cc1 -std=c++2a -include-pch %t-cxx2a %s -ast-print -verify | FileCheck %s

#ifndef USE_PCH
namespace inheriting_constructor {
  struct S {};

  template<typename X, typename Y> struct T {
    template<typename A>
    explicit(((void)Y{}, true)) T(A &&a) {}
  };

  template<typename X, typename Y> struct U : T<X, Y> {
    using T<X, Y>::T;
  };

  U<S, char> foo(char ch) {
    return U<S, char>(ch);
  }
}
#else
namespace inheriting_constructor {
U<S, char> a = foo('0');
}

//CHECK: explicit(((void)char{} , true))

#endif

namespace basic {
#ifndef USE_PCH

struct B {};

struct A {
  explicit A(int);
  explicit(false) operator bool();
  explicit(true) operator B();
};
#else
//expected-note@-6+ {{candidate constructor}}
//expected-note@-9+ {{candidate constructor}}
//expected-note-re@-7+ {{explicit constructor is not a candidate{{$}}}}
//expected-note@-7+ {{candidate function}}
//expected-note@-7+ {{explicit conversion function is not a candidate (explicit specifier evaluates to true)}}

//CHECK: explicit{{ +}}A(
//CHECK-NEXT: explicit(false){{ +}}operator
//CHECK-NEXT: explicit(true){{ +}}operator
A a = 0; //expected-error {{no viable conversion}}
A a1(0);

bool b = a1;
B b1 = a1; //expected-error {{no viable conversion}}

#endif
}


namespace templ {
#ifndef USE_PCH

template<bool b>
struct B {
  static constexpr bool value = b;
};

template<bool b>
struct A {
  explicit(b) A(B<b>) {}
  template<typename T>
  explicit(b ^ T::value) operator T();
};
B<true> b_true;
B<false> b_false;
#else
//expected-note@-8 {{candidate template ignored}} expected-note@-8 {{implicit deduction guide declared as 'template <bool b> A(templ::A<b>) -> templ::A<b>'}}
//expected-note@-8 {{explicit constructor declared here}} expected-note@-8 {{implicit deduction guide declared as 'template <bool b> explicit(b) A(B<b>) -> templ::A<b>'}}
//expected-note@-15+ {{candidate constructor}}
//expected-note@-8+ {{explicit conversion function is not a candidate (explicit specifier}}
//expected-note@-11 {{explicit constructor is not a candidate (explicit specifier}}

//CHECK: explicit(b){{ +}}A
//CHECK: explicit(b{{ +}}^{{ +}}T::value){{ +}}operator

A a = { b_true }; //expected-error {{class template argument deduction}}
A a0 = b_true; //expected-error {{no viable constructor or deduction guide}}
A a_true(b_true);
A a_false = b_false;

B<true> b = a_true;
B<true> b1 = a_false; //expected-error {{no viable conversion}}
B<false> b2(a_true);

#endif

}

namespace guide {

#ifndef USE_PCH

template<typename T>
struct A {
  A(T);
};

template<typename T>
explicit(true) A(T) -> A<T>;

explicit(false) A(int) -> A<int>;

#else
//expected-note@-5 {{explicit deduction guide}}

//CHECK: explicit(true){{ +}}A(
//CHECK: explicit(false){{ +}}A(

A a = { 0.0 }; //expected-error {{explicit deduction guide}}
A a1 = { 0 };

#endif

}

#define USE_PCH
