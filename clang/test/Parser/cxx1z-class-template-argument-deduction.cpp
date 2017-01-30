// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -verify %s

template<typename T> struct A {}; // expected-note 35{{declared here}}

// Make sure we still correctly parse cases where a template can appear without arguments.
namespace template_template_arg {
  template<template<typename> typename> struct X {};
  template<typename> struct Y {};

  X<A> xa;
  Y<A> ya; // expected-error {{requires template arguments}}
  X<::A> xcca;
  Y<::A> ycca; // expected-error {{requires template arguments}}

  template<template<typename> typename = A> struct XD {};
  template<typename = A> struct YD {}; // expected-error {{requires template arguments}}
  template<template<typename> typename = ::A> struct XCCD {};
  template<typename = ::A> struct YCCD {}; // expected-error {{requires template arguments}}

  // FIXME: replacing the invalid type with 'int' here is horrible
  template <A a = A<int>()> class C { }; // expected-error {{requires template arguments}} expected-error {{not implicitly convertible to 'int'}}
  template<typename T = A> struct G { }; // expected-error {{requires template arguments}}
}

namespace injected_class_name {
  template<typename T> struct A {
    A(T);
    void f(int) {
      A a = 1;
      injected_class_name::A b = 1; // expected-error {{not yet supported}}
    }
    void f(T);
  };
  A<short> ai = 1;
  A<double>::A b(1); // expected-error {{constructor name}}
}

struct member {
  A a; // expected-error {{requires template arguments}}
  A *b; // expected-error {{requires template arguments}}
  const A c; // expected-error {{requires template arguments}}

  void f() throw (A); // expected-error {{requires template arguments}}

  friend A; // expected-error {{requires template arguments; argument deduction not allowed in friend declaration}}

  operator A(); // expected-error {{requires template arguments; argument deduction not allowed in conversion function type}}

  static A x; // expected-error {{requires an initializer}}
  static A y = 0; // expected-error {{not yet supported}}
};

namespace in_typedef {
  typedef A *AutoPtr; // expected-error {{requires template arguments; argument deduction not allowed in typedef}}
  typedef A (*PFun)(int a); // expected-error{{requires template arguments; argument deduction not allowed in typedef}}
  typedef A Fun(int a) -> decltype(a + a); // expected-error{{requires template arguments; argument deduction not allowed in function return type}}
}

namespace stmt {
  void g(A a) { // expected-error{{requires template arguments; argument deduction not allowed in function prototype}}
    try { }
    catch (A &a) { } // expected-error{{requires template arguments; argument deduction not allowed in exception declaration}}
    catch (const A a) { } // expected-error{{requires template arguments; argument deduction not allowed in exception declaration}}
    try { } catch (A a) { } // expected-error{{requires template arguments; argument deduction not allowed in exception declaration}}

    // FIXME: The standard only permits class template argument deduction in a
    // simple-declaration or cast. We also permit it in conditions,
    // for-range-declarations, member-declarations for static data members, and
    // new-expressions, because not doing so would be bizarre.
    A local = 0; // expected-error {{not yet supported}}
    static A local_static = 0; // expected-error {{not yet supported}}
    static thread_local A thread_local_static = 0; // expected-error {{not yet supported}}
    if (A a = 0) {} // expected-error {{not yet supported}}
    if (A a = 0; a) {} // expected-error {{not yet supported}}
    switch (A a = 0) {} // expected-error {{not yet supported}}
    switch (A a = 0; a) {} // expected-error {{not yet supported}}
    for (A a = 0; a; /**/) {} // expected-error {{not yet supported}}
    for (/**/; A a = 0; /**/) {} // expected-error {{not yet supported}}
    while (A a = 0) {} // expected-error {{not yet supported}}
    int arr[3];
    for (A a : arr) {} // expected-error {{not yet supported}}
  }

  namespace std {
    class type_info;
  }
}

namespace expr {
  template<typename T> struct U {};
  void j() {
    (void)typeid(A); // expected-error{{requires template arguments; argument deduction not allowed here}}
    (void)sizeof(A); // expected-error{{requires template arguments; argument deduction not allowed here}}
    (void)__alignof(A); // expected-error{{requires template arguments; argument deduction not allowed here}}

    U<A> v; // expected-error {{requires template arguments}}

    int n;
    (void)dynamic_cast<A&>(n); // expected-error{{requires template arguments; argument deduction not allowed here}}
    (void)static_cast<A*>(&n); // expected-error{{requires template arguments; argument deduction not allowed here}}
    (void)reinterpret_cast<A*>(&n); // expected-error{{requires template arguments; argument deduction not allowed here}}
    (void)const_cast<A>(n); // expected-error{{requires template arguments; argument deduction not allowed here}}
    (void)*(A*)(&n); // expected-error{{requires template arguments; argument deduction not allowed here}}
    (void)(A)(n); // expected-error{{requires template arguments; argument deduction not allowed here}}
    (void)(A){n}; // expected-error{{requires template arguments; argument deduction not allowed here}}

    (void)A(n); // expected-error {{not yet supported}}
    (void)A{n}; // expected-error {{not yet supported}}
    (void)new A(n); // expected-error {{not yet supported}}
    (void)new A{n}; // expected-error {{not yet supported}}
    // FIXME: We should diagnose the lack of an initializer here.
    (void)new A; // expected-error {{not yet supported}}
  }
}

namespace decl {
  enum E : A {}; // expected-error{{requires template arguments; argument deduction not allowed here}}
  struct F : A {}; // expected-error{{expected class name}}

  using B = A; // expected-error{{requires template arguments}}

  auto k() -> A; // expected-error{{requires template arguments}}

  A a; // expected-error {{requires an initializer}}
  A b = 0; // expected-error {{not yet supported}}
  const A c = 0; // expected-error {{not yet supported}}
  A (parens) = 0; // expected-error {{cannot use parentheses when declaring variable with deduced class template specialization type}}
  A *p = 0; // expected-error {{cannot form pointer to deduced class template specialization type}}
  A &r = *p; // expected-error {{cannot form reference to deduced class template specialization type}}
  A arr[3] = 0; // expected-error {{cannot form array of deduced class template specialization type}}
  A F::*pm = 0; // expected-error {{cannot form pointer to deduced class template specialization type}}
  A (*fp)() = 0; // expected-error {{cannot form function returning deduced class template specialization type}}
  A [x, y] = 0; // expected-error {{cannot be declared with type 'A'}} expected-error {{not yet supported}}
}

namespace typename_specifier {
  struct F {};

  void e() {
    (void) typename ::A(0); // expected-error {{not yet supported}}
    (void) typename ::A{0}; // expected-error {{not yet supported}}
    new typename ::A(0); // expected-error {{not yet supported}}
    new typename ::A{0}; // expected-error {{not yet supported}}
    typename ::A a = 0; // expected-error {{not yet supported}}
    const typename ::A b = 0; // expected-error {{not yet supported}}
    if (typename ::A a = 0) {} // expected-error {{not yet supported}}
    for (typename ::A a = 0; typename ::A b = 0; /**/) {} // expected-error 2{{not yet supported}}

    (void)(typename ::A)(0); // expected-error{{requires template arguments; argument deduction not allowed here}}
    (void)(typename ::A){0}; // expected-error{{requires template arguments; argument deduction not allowed here}}
  }
  typename ::A a = 0; // expected-error {{not yet supported}}
  const typename ::A b = 0; // expected-error {{not yet supported}}
  typename ::A (parens) = 0; // expected-error {{cannot use parentheses when declaring variable with deduced class template specialization type}}
  typename ::A *p = 0; // expected-error {{cannot form pointer to deduced class template specialization type}}
  typename ::A &r = *p; // expected-error {{cannot form reference to deduced class template specialization type}}
  typename ::A arr[3] = 0; // expected-error {{cannot form array of deduced class template specialization type}}
  typename ::A F::*pm = 0; // expected-error {{cannot form pointer to deduced class template specialization type}}
  typename ::A (*fp)() = 0; // expected-error {{cannot form function returning deduced class template specialization type}}
  typename ::A [x, y] = 0; // expected-error {{cannot be declared with type 'typename ::A'}} expected-error {{not yet supported}}

  struct X { template<typename T> struct A {}; }; // expected-note 8{{template}}

  template<typename T> void f() {
    (void) typename T::A(0); // expected-error {{not yet supported}}
    (void) typename T::A{0}; // expected-error {{not yet supported}}
    new typename T::A(0); // expected-error {{not yet supported}}
    new typename T::A{0}; // expected-error {{not yet supported}}
    typename T::A a = 0; // expected-error {{not yet supported}}
    const typename T::A b = 0; // expected-error {{not yet supported}}
    if (typename T::A a = 0) {} // expected-error {{not yet supported}}
    for (typename T::A a = 0; typename T::A b = 0; /**/) {} // expected-error 2{{not yet supported}}

    {(void)(typename T::A)(0);} // expected-error{{refers to class template member}}
    {(void)(typename T::A){0};} // expected-error{{refers to class template member}}
    {typename T::A (parens) = 0;} // expected-error {{refers to class template member in 'typename_specifier::X'; argument deduction not allowed here}}
    {typename T::A *p = 0;} // expected-error {{refers to class template member}}
    {typename T::A &r = *p;} // expected-error {{refers to class template member}}
    {typename T::A arr[3] = 0;} // expected-error {{refers to class template member}}
    {typename T::A F::*pm = 0;} // expected-error {{refers to class template member}}
    {typename T::A (*fp)() = 0;} // expected-error {{refers to class template member}}
    {typename T::A [x, y] = 0;} // expected-error {{cannot be declared with type 'typename T::A'}} expected-error {{not yet supported}}
  }
  template void f<X>(); // expected-note {{instantiation of}}

  template<typename T> void g(typename T::A = 0); // expected-note {{refers to class template member}}
  void h() { g<X>(); } // expected-error {{no matching function}}
}
