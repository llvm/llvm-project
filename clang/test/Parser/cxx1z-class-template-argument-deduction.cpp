// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -verify %s

template <typename T> struct A { // expected-note 35{{declared here}}
  constexpr A() {}
  constexpr A(int) {}
  constexpr operator int() { return 0; }
};
A() -> A<int>;
A(int) -> A<int>;

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
  template <A a = A<int>()> class C { }; // expected-error {{requires template arguments}}
  template<typename T = A> struct G { }; // expected-error {{requires template arguments}}
}

namespace injected_class_name {
  template<typename T> struct A {
    A(T);
    void f(int) { // expected-note {{previous}}
      A a = 1;
      injected_class_name::A b = 1; // expected-note {{in instantiation of template class 'injected_class_name::A<int>'}}
    }
    void f(T); // expected-error {{multiple overloads of 'f' instantiate to the same signature 'void (int)}}
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

  static A x; // expected-error {{declaration of variable 'x' with deduced type 'A' requires an initializer}}
  static constexpr A y = 0;
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
    A local = 0;
    static A local_static = 0;
    static thread_local A thread_local_static = 0;
    if (A a = 0) {}
    if (A a = 0; a) {}
    switch (A a = 0) {} // expected-warning {{no case matching constant switch condition '0'}}
    switch (A a = 0; a) {} // expected-warning {{no case matching constant switch condition '0'}}
    for (A a = 0; a; /**/) {}
    for (/**/; A a = 0; /**/) {}
    while (A a = 0) {}
    int arr[3];
    for (A a : arr) {}
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

    (void)A(n);
    (void)A{n};
    (void)new A(n);
    (void)new A{n};
    // FIXME: We should diagnose the lack of an initializer here.
    (void)new A;
  }
}

namespace decl {
  enum E : A {}; // expected-error{{requires template arguments; argument deduction not allowed here}}
  struct F : A {}; // expected-error{{expected class name}}

  using B = A; // expected-error{{requires template arguments}}

  auto k() -> A; // expected-error{{requires template arguments}}

  A a; // expected-error {{declaration of variable 'a' with deduced type 'A' requires an initializer}}
  A b = 0;
  const A c = 0;
  A (parens) = 0; // expected-error {{cannot use parentheses when declaring variable with deduced class template specialization type}}
  A *p = 0; // expected-error {{cannot form pointer to deduced class template specialization type}}
  A &r = *p; // expected-error {{cannot form reference to deduced class template specialization type}}
  A arr[3] = 0; // expected-error {{cannot form array of deduced class template specialization type}}
  A F::*pm = 0; // expected-error {{cannot form pointer to deduced class template specialization type}}
  A (*fp)() = 0; // expected-error {{cannot form function returning deduced class template specialization type}}
  A [x, y] = 0; // expected-error {{cannot be declared with type 'A'}} expected-error {{type 'A<int>' decomposes into 0 elements, but 2 names were provided}}
}

namespace typename_specifier {
  struct F {};

  void e() {
    (void) typename ::A(0);
    (void) typename ::A{0};
    new typename ::A(0);
    new typename ::A{0};
    typename ::A a = 0;
    const typename ::A b = 0;
    if (typename ::A a = 0) {}
    for (typename ::A a = 0; typename ::A b = 0; /**/) {}

    (void)(typename ::A)(0); // expected-error{{requires template arguments; argument deduction not allowed here}}
    (void)(typename ::A){0}; // expected-error{{requires template arguments; argument deduction not allowed here}}
  }
  typename ::A a = 0;
  const typename ::A b = 0;
  typename ::A (parens) = 0; // expected-error {{cannot use parentheses when declaring variable with deduced class template specialization type}}
  typename ::A *p = 0; // expected-error {{cannot form pointer to deduced class template specialization type}}
  typename ::A &r = *p; // expected-error {{cannot form reference to deduced class template specialization type}}
  typename ::A arr[3] = 0; // expected-error {{cannot form array of deduced class template specialization type}}
  typename ::A F::*pm = 0; // expected-error {{cannot form pointer to deduced class template specialization type}}
  typename ::A (*fp)() = 0; // expected-error {{cannot form function returning deduced class template specialization type}}
  typename ::A [x, y] = 0; // expected-error {{cannot be declared with type 'typename ::A'}} expected-error {{type 'typename ::A<int>' (aka 'A<int>') decomposes into 0}}

  struct X { template<typename T> struct A { A(T); }; }; // expected-note 8{{declared here}}

  template<typename T> void f() {
    (void) typename T::A(0);
    (void) typename T::A{0};
    new typename T::A(0);
    new typename T::A{0};
    typename T::A a = 0;
    const typename T::A b = 0;
    if (typename T::A a = 0) {} // expected-error {{value of type 'typename X::A<int>' (aka 'typename_specifier::X::A<int>') is not contextually convertible to 'bool'}}
    for (typename T::A a = 0; typename T::A b = 0; /**/) {} // expected-error {{value of type 'typename X::A<int>' (aka 'typename_specifier::X::A<int>') is not contextually convertible to 'bool'}}

    {(void)(typename T::A)(0);} // expected-error{{refers to class template member}}
    {(void)(typename T::A){0};} // expected-error{{refers to class template member}}
    {typename T::A (parens) = 0;} // expected-error {{refers to class template member in 'typename_specifier::X'; argument deduction not allowed here}}
    // expected-warning@-1 {{disambiguated as redundant parentheses around declaration of variable named 'parens'}} expected-note@-1 {{add a variable name}} expected-note@-1{{remove parentheses}} expected-note@-1 {{add enclosing parentheses}}
    {typename T::A *p = 0;} // expected-error {{refers to class template member}}
    {typename T::A &r = *p;} // expected-error {{refers to class template member}}
    {typename T::A arr[3] = 0;} // expected-error {{refers to class template member}}
    {typename T::A F::*pm = 0;} // expected-error {{refers to class template member}}
    {typename T::A (*fp)() = 0;} // expected-error {{refers to class template member}}
    {typename T::A [x, y] = 0;} // expected-error {{cannot be declared with type 'typename T::A'}} expected-error {{type 'typename X::A<int>' (aka 'typename_specifier::X::A<int>') decomposes into 0}}
  }
  template void f<X>(); // expected-note {{instantiation of}}

  template<typename T> void g(typename T::A = 0); // expected-note {{refers to class template member}}
  void h() { g<X>(); } // expected-error {{no matching function}}
}
