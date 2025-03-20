// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// Clang used to crash trying to recover while adding 'this->' before Work(x);

template <typename> struct A {
  static void Work(int);  // expected-note{{here}}
};

template <typename T> struct B : public A<T> {
  template <typename T2> B(T2 x) {
    Work(x);  // expected-error{{explicit qualification required}}
  }
};

void Test() {
  B<int> b(0);  // expected-note{{in instantiation of function template}}
}


// Don't crash here.
namespace PR16134 {
  template <class P> struct S // expected-error {{expected ';'}}
  template <> static S<Q>::f() // expected-error +{{}}
}

namespace PR16225 {
  template <typename T> void f();
  template <typename C> void g(C*) {
    struct LocalStruct : UnknownBase<Mumble, C> { };  // expected-error {{use of undeclared identifier 'Mumble'}}
    f<LocalStruct>();
#if __cplusplus <= 199711L
    // expected-warning@-2 {{template argument uses local type 'LocalStruct'}}
    // expected-note@-3 {{while substituting explicitly-specified template arguments}}
#endif
    struct LocalStruct2 : UnknownBase<C> { };  // expected-error {{no template named 'UnknownBase'}}
  }
  struct S;
  void h() {
    g<S>(0);
#if __cplusplus <= 199711L
    // expected-note@-2 {{in instantiation of function template specialization}}
#endif
  }
}

namespace test1 {
  template <typename> class ArraySlice {};
  class Foo;
  class NonTemplateClass { // #defined-here
    void MemberFunction(ArraySlice<Foo>, int);
    template <class T> void MemberFuncTemplate(ArraySlice<T>, int);
  };
  void NonTemplateClass::MemberFunction(ArraySlice<Foo> resource_data,
                                        int now) {
    // expected-note@+1 {{in instantiation of function template specialization 'test1::NonTemplateClass::MemberFuncTemplate<test1::Foo>'}}
    MemberFuncTemplate(resource_data, now);
  }
  template <class T>
  void NonTemplateClass::MemberFuncTemplate(ArraySlice<T> resource_data, int) {
    // expected-error@+1 {{member 'UndeclaredMethod' used before its declaration}}
    UndeclaredMethod(resource_data);
  }
  // expected-error@+3 {{out-of-line definition of 'UndeclaredMethod' does not match any declaration}}
  // expected-note@+2 {{member is declared here}}
  // expected-note@#defined-here {{defined here}}
  void NonTemplateClass::UndeclaredMethod() {}
}
