// RUN: %clang_cc1 -fsyntax-only -verify -triple=x86_64-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple=x86_64-linux-gnu -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple=x86_64-linux-gnu -std=c++11 %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -triple=x86_64-linux-gnu %s -DCPP11ONLY

// C++11 [temp.arg.nontype]p1:
//
//   A template-argument for a non-type, non-template template-parameter shall
//   be one of:
//   -- an integral constant expression; or
//   -- the name of a non-type template-parameter ; or
#ifndef CPP11ONLY 

namespace non_type_tmpl_param {
  template <int N> struct X0 { X0(); };
  template <int N> X0<N>::X0() { }
  template <int* N> struct X1 { X1(); };
  template <int* N> X1<N>::X1() { }
  template <int& N> struct X3 { X3(); };
  template <int& N> X3<N>::X3() { }
  template <int (*F)(int)> struct X4 { X4(); };
  template <int (*F)(int)> X4<F>::X4() { }
  template <typename T, int (T::* M)(int)> struct X5 { X5(); };
  template <typename T, int (T::* M)(int)> X5<T, M>::X5() { }
}

//   -- a constant expression that designates the address of an object with
//      static storage duration and external or internal linkage or a function
//      with external or internal linkage, including function templates and
//      function template-ids, but excluting non-static class members, expressed
//      (ignoring parentheses) as & id-expression, except that the & may be
//      omitted if the name refers to a function or array and shall be omitted
//      if the corresopnding template-parameter is a reference; or
namespace addr_of_obj_or_func {
  template <int* p> struct X0 { }; // expected-note 5{{here}}
#if __cplusplus >= 201103L
  // expected-note@-2 2{{template parameter is declared here}}
#endif

  template <int (*fp)(int)> struct X1 { };
  template <int &p> struct X2 { }; // expected-note 4{{here}}
  template <const int &p> struct X2k { }; // expected-note {{here}}
  template <int (&fp)(int)> struct X3 { }; // expected-note 4{{here}}

  int i = 42;
#if __cplusplus >= 201103L
  // expected-note@-2 {{declared here}}
#endif

  int iarr[10];
  int f(int i);
  const int ki = 9;
#if __cplusplus <= 199711L
  // expected-note@-2 5{{non-type template argument refers to object here}}
#endif

  __thread int ti = 100; // expected-note {{here}}
#if __cplusplus <= 199711L
  // expected-note@-2 {{here}}
#endif

  static int f_internal(int);
#if __cplusplus <= 199711L
  // expected-note@-2 4{{non-type template argument refers to function here}}
#endif

  template <typename T> T f_tmpl(T t);
  struct S { union { int NonStaticMember; }; };

  void test() {
    X0<i> x0a;
#if __cplusplus <= 199711L
    // expected-error@-2 {{non-type template argument for template parameter of pointer type 'int *' must have its address taken}}
#else
    // expected-error@-4 {{non-type template argument of type 'int' is not a constant expression}}
    // expected-note@-5 {{read of non-const variable 'i' is not allowed in a constant expression}}
#endif
    X0<&i> x0a_addr;
    X0<iarr> x0b;
    X0<&iarr> x0b_addr; // expected-error {{cannot be converted to a value of type 'int *'}}
    X0<ki> x0c; // expected-error {{must have its address taken}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{internal linkage is a C++11 extension}}
#endif

    X0<&ki> x0c_addr; // expected-error {{cannot be converted to a value of type 'int *'}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{internal linkage is a C++11 extension}}
#endif

    X0<&ti> x0d_addr;
#if __cplusplus <= 199711L
    // expected-error@-2 {{non-type template argument refers to thread-local object}}
#else
    // expected-error@-4 {{non-type template argument of type 'int *' is not a constant expression}}
#endif

    X1<f> x1a;
    X1<&f> x1a_addr;
    X1<f_tmpl> x1b;
    X1<&f_tmpl> x1b_addr;
    X1<f_tmpl<int> > x1c;
    X1<&f_tmpl<int> > x1c_addr;
    X1<f_internal> x1d;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{internal linkage is a C++11 extension}}
#endif

    X1<&f_internal> x1d_addr;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{internal linkage is a C++11 extension}}
#endif

    X2<i> x2a;
    X2<&i> x2a_addr; // expected-error {{address taken}}
    X2<iarr> x2b; // expected-error {{cannot bind to template argument of type 'int[10]'}}
    X2<&iarr> x2b_addr; // expected-error {{address taken}}
    X2<ki> x2c; // expected-error {{ignores qualifiers}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{internal linkage is a C++11 extension}}
#endif

    X2k<ki> x2kc;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{internal linkage is a C++11 extension}}
#endif

    X2k<&ki> x2kc_addr; // expected-error {{address taken}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{internal linkage is a C++11 extension}}
#endif

    X2<ti> x2d_addr; // expected-error {{refers to thread-local object}}
    X3<f> x3a;
    X3<&f> x3a_addr; // expected-error {{address taken}}
    X3<f_tmpl> x3b;
    X3<&f_tmpl> x3b_addr; // expected-error {{address taken}}
    X3<f_tmpl<int> > x3c;
    X3<&f_tmpl<int> > x3c_addr; // expected-error {{address taken}}
    X3<f_internal> x3d;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{internal linkage is a C++11 extension}}
#endif

    X3<&f_internal> x3d_addr; // expected-error {{address taken}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{internal linkage is a C++11 extension}}
#endif

    int n;
#if __cplusplus <= 199711L
    // expected-note@-2 {{non-type template argument refers to object here}}
#else
    // expected-note@-4 {{declared here}}
#endif

    X0<&n> x0_no_linkage;
#if __cplusplus <= 199711L
    // expected-error@-2 {{non-type template argument refers to object 'n' that does not have linkage}}
#else
    // expected-error@-4 {{non-type template argument of type 'int *' is not a constant expression}}
    // expected-note@-5 {{pointer to 'n' is not a constant expression}}
#endif

    struct Local { static int f() {} }; // expected-note {{here}}
    X1<&Local::f> x1_no_linkage; // expected-error {{non-type template argument refers to function 'f' that does not have linkage}}
    X0<&S::NonStaticMember> x0_non_static; // expected-error {{non-static data member}}
  }
}

//   -- a constant expression that evaluates to a null pointer value (4.10); or
//   -- a constant expression that evaluates to a null member pointer value
//      (4.11); or
//   -- a pointer to member expressed as described in 5.3.1.

namespace bad_args {
  template <int* N> struct X0 { }; // expected-note 2{{template parameter is declared here}}
  int i = 42;
  X0<&i + 2> x0a; // expected-error{{non-type template argument does not refer to any declaration}}
  int* iptr = &i;
#if __cplusplus >= 201103L
  // expected-note@-2 {{declared here}}
#endif

  X0<iptr> x0b;
#if __cplusplus <= 199711L
  // expected-error@-2 {{non-type template argument for template parameter of pointer type 'int *' must have its address taken}}
#else
  // expected-error@-4 {{non-type template argument of type 'int *' is not a constant expression}}
  // expected-note@-5 {{read of non-constexpr variable 'iptr' is not allowed in a constant expression}}
#endif
}
#endif // CPP11ONLY

namespace default_args {
#ifdef CPP11ONLY
namespace lambdas {
template<int I = ([] { return 5; }())> //expected-error {{constant expression}}
int f();
}
#endif // CPP11ONLY

}
