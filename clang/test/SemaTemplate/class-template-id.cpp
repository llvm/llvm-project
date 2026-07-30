// RUN: %clang_cc1 -fsyntax-only -verify=expected,precxx23,precxx17 %std_cxx98-14 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,precxx23,cxx17 %std_cxx17-20 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx17,cxx23 %std_cxx23- %s
template<typename T, typename U = float> struct A { };

typedef A<int> A_int;

typedef float FLOAT;

A<int, FLOAT> *foo(A<int> *ptr, A<int> const *ptr2, A<int, double> *ptr3) {
  if (ptr)
    return ptr; // okay
  else if (ptr2)
    return ptr2; // precxx23-error{{cannot initialize return object of type 'A<int, FLOAT> *' (aka 'A<int, float> *') with an lvalue of type 'const A<int> *'}} \
                    cxx23-error{{cannot initialize return object of type 'A<int, FLOAT> *' (aka 'A<int, float> *') with an rvalue of type 'const A<int> *'}}
  else {
    return ptr3; // precxx23-error{{cannot initialize return object of type 'A<int, FLOAT> *' (aka 'A<int, float> *') with an lvalue of type 'A<int, double> *'}} \
                    cxx23-error{{cannot initialize return object of type 'A<int, FLOAT> *' (aka 'A<int, float> *') with an rvalue of type 'A<int, double> *'}}
  }
}

template<int I> struct B;

const int value = 12;
B<17 + 2> *bar(B<(19)> *ptr1, B< (::value + 7) > *ptr2, B<19 - 3> *ptr3) {
  if (ptr1)
    return ptr1;
  else if (ptr2)
    return ptr2;
  else
    return ptr3; // precxx23-error{{cannot initialize return object of type 'B<17 + 2> *' with an lvalue of type 'B<19 - 3>}} \
                    cxx23-error{{cannot initialize return object of type 'B<17 + 2> *' with an rvalue of type 'B<19 - 3>}}
}

typedef B<5> B5;


namespace N {
  template<typename T> struct C {};
}

N::C<int> c1;
typedef N::C<float> c2;

// PR5655
template<typename T> struct Foo { }; // precxx17-note {{template is declared here}} \
                                        cxx17-note {{candidate template ignored: couldn't infer template argument 'T'}} \
                                        cxx17-note {{implicit deduction guide declared as 'template <typename T> Foo() -> Foo<T>'}} \
                                        cxx17-note {{implicit deduction guide declared as 'template <typename T> Foo(Foo<T>) -> Foo<T>'}} \
                                        cxx17-note {{candidate function template not viable: requires 1 argument, but 0 were provided}}

void f(void) { Foo bar; } // precxx17-error {{use of class template 'Foo' requires template arguments}} \
                             cxx17-error {{no viable constructor or deduction guide for deduction of template arguments of 'Foo'}}

template <typename T> class Party;
template <> class Party<T> { friend struct Party<>; }; // expected-error {{use of undeclared identifier 'T'}}
