// RUN: %clang_cc1 -std=c++11 -fms-compatibility -fsyntax-only -verify %s

// If we were even more clever, we'd tell the user to use one set of parens to
// get the size of this type, so they don't get errors after inserting typename.

namespace basic {
template <typename T> int type_f() { return sizeof T::type; }  // expected-error {{missing 'typename' prior to dependent type name 'X::type'}}
template <typename T> int type_g() { return sizeof(T::type); }  // expected-warning {{missing 'typename' prior to dependent type name 'X::type'}}
template <typename T> int type_h() { return sizeof((T::type)); }  // expected-error {{missing 'typename' prior to dependent type name 'X::type'}}
template <typename T> int value_f() { return sizeof T::not_a_type; }
template <typename T> int value_g() { return sizeof(T::not_a_type); }
template <typename T> int value_h() { return sizeof((T::not_a_type)); }
struct X {
  typedef int type;
  static const int not_a_type;
};
int bar() {
  return
      type_f<X>() + // expected-note-re {{in instantiation {{.*}} requested here}}
      type_g<X>() + // expected-note-re {{in instantiation {{.*}} requested here}}
      type_h<X>() + // expected-note-re {{in instantiation {{.*}} requested here}}
      value_f<X>() +
      value_f<X>() +
      value_f<X>();
}
}

namespace nested_sizeof {
template <typename T>
struct Foo {
  enum {
    // expected-warning@+2 {{use 'template' keyword to treat 'InnerTemplate' as a dependent template name}}
    // expected-warning@+1 {{missing 'typename' prior to dependent type name 'Bar::InnerType'}}
    x1 = sizeof(typename T::/*template*/ InnerTemplate<sizeof(/*typename*/ T::InnerType)>),
    // expected-warning@+1 {{missing 'typename' prior to dependent type name 'Bar::InnerType'}}
    x2 = sizeof(typename T::template InnerTemplate<sizeof(/*typename*/ T::InnerType)>),
    // expected-warning@+1 {{use 'template' keyword to treat 'InnerTemplate' as a dependent template name}}
    y1 = sizeof(typename T::/*template*/ InnerTemplate<sizeof(T::InnerVar)>),
    y2 = sizeof(typename T::template InnerTemplate<sizeof(T::InnerVar)>),
    z = sizeof(T::template InnerTemplate<sizeof(T::InnerVar)>::x),
  };
};
struct Bar {
  template <int N>
  struct InnerTemplate { int x[N]; };
  typedef double InnerType;
  static const int InnerVar = 42;
};
template struct Foo<Bar>; // expected-note-re {{in instantiation {{.*}} requested here}}
}

namespace ambiguous_missing_parens {
// expected-error@+1 {{'Q::U' is expected to be a non-type template, but instantiated to a class template}}
template <typename T> void f() { int a = sizeof T::template U<0> + 4; }
struct Q {
  // expected-note@+1 {{class template declared here}}
  template <int> struct U {};
};
// expected-note-re@+1 {{in instantiation {{.*}} requested here}}
template void f<Q>();
}
