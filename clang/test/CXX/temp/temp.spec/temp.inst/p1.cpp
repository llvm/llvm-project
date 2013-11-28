// RUN: %clang_cc1 -std=c++11 -verify %s

// The implicit specialization of a class template specialuzation causes the
// implicit instantiation of the declarations, but not the definitions or
// default arguments, of:

// FIXME: Many omitted cases

// - scoped member enumerations
namespace ScopedEnum {
  template<typename T> struct ScopedEnum1 {
    enum class E {
      e = T::error // expected-error {{'double' cannot be used prior to '::'}}
    };
  };
  ScopedEnum1<int> se1; // ok

  template<typename T> struct ScopedEnum2 {
    enum class E : T { // expected-error {{non-integral type 'void *' is an invalid underlying type}}
      e = 0
    };
  };
  ScopedEnum2<void*> se2; // expected-note {{here}}

  template<typename T> struct UnscopedEnum3 {
    enum class E : T {
      e = 4
    };
    int arr[(int)E::e];
  };
  UnscopedEnum3<int> ue3; // ok

  ScopedEnum1<double>::E e1; // ok
  ScopedEnum1<double>::E e2 = decltype(e2)::e; // expected-note {{in instantiation of enumeration 'ScopedEnum::ScopedEnum1<double>::E' requested here}}

  // DR1484 specifies that enumerations cannot be separately instantiated,
  // they will be instantiated with the rest of the template declaration.
  template<typename T>
  int f() {
    enum class E {
      e = T::error // expected-error {{has no members}}
    };
    return (int)E();
  }
  int test1 = f<int>(); // expected-note {{here}}

  template<typename T>
  int g() {
    enum class E {
      e = T::error // expected-error {{has no members}}
    };
    return E::e;
  }
  int test2 = g<int>(); // expected-note {{here}}
}

// And it cases the implicit instantiations of the definitions of:

// - unscoped member enumerations
namespace UnscopedEnum {
  template<typename T> struct UnscopedEnum1 {
    enum E {
      e = T::error // expected-error {{'int' cannot be used prior to '::'}}
    };
  };
  UnscopedEnum1<int> ue1; // expected-note {{here}}

  template<typename T> struct UnscopedEnum2 {
    enum E : T { // expected-error {{non-integral type 'void *' is an invalid underlying type}}
      e = 0
    };
  };
  UnscopedEnum2<void*> ue2; // expected-note {{here}}

  template<typename T> struct UnscopedEnum3 {
    enum E : T {
      e = 4
    };
    int arr[E::e];
  };
  UnscopedEnum3<int> ue3; // ok

  template<typename T>
  int f() {
    enum E {
      e = T::error // expected-error {{has no members}}
    };
    return (int)E();
  }
  int test1 = f<int>(); // expected-note {{here}}

  template<typename T>
  int g() {
    enum E {
      e = T::error // expected-error {{has no members}}
    };
    return E::e;
  }
  int test2 = g<int>(); // expected-note {{here}}
}

// FIXME:
//- - member anonymous unions
