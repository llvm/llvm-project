// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR6915 {
  template <typename T>
  class D {
    enum T::X v; // expected-error{{use of 'X' with tag type that does not match previous declaration}} \
    // expected-error{{no enum named 'X' in 'PR6915::D3'}}
  };

  struct D1 {
    enum X { value };
  };
  struct D2 {
    class X { }; // expected-note{{previous use is here}}
  };
  struct D3 { };

  template class D<D1>;
  template class D<D2>; // expected-note{{in instantiation of}}
  template class D<D3>; // expected-note{{in instantiation of}}
}

template<typename T>
struct DeclOrDef {
  enum T::foo; // expected-error{{nested name specifier for a declaration cannot depend on a template parameter}}
               // expected-error@-1{{forward declaration of enum cannot have a nested name specifier}}
  enum T::bar { // expected-error{{nested name specifier for a declaration cannot depend on a template parameter}}
    value
  };
};

namespace PR6649 {
  template <typename T> struct foo {
    class T::bar;  // expected-error{{nested name specifier for a declaration cannot depend on a template parameter}}
                   // expected-error@-1{{forward declaration of class cannot have a nested name specifier}}
    class T::bar { int x; }; // expected-error{{nested name specifier for a declaration cannot depend on a template parameter}}
  };
}

namespace rdar8568507 {
  template <class T> struct A *makeA(T t);
}

namespace canon {
  template <class T> void t1(struct T::X) {}
  // expected-note@-1 {{previous definition is here}}
  template <class T> void t1(class T::X) {}
  // expected-error@-1 {{redefinition of 't1'}}

  template <class T> void t2(struct T::template X<int>) {}
  // expected-note@-1 {{previous definition is here}}
  template <class T> void t2(class T::template X<int>) {}
  // expected-error@-1 {{redefinition of 't2'}}

  template <class T> constexpr int t3(typename T::X* = 0) { return 0; } // #canon-t3-0
  template <class T> constexpr int t3(struct   T::X* = 0) { return 1; } // #canon-t3-1
  template <class T> constexpr int t3(union    T::X* = 0) { return 2; } // #canon-t3-2
  template <class T> constexpr int t3(enum     T::X* = 0) { return 3; } // #canon-t3-3

  struct A { using X = int; };
  static_assert(t3<A>() == 0);

  struct B { struct X {}; };
  static_assert(t3<B>() == 1);
  // expected-error@-1 {{call to 't3' is ambiguous}}
  // expected-note@#canon-t3-0 {{candidate function}}
  // expected-note@#canon-t3-1 {{candidate function}}

  struct C { union X {}; };
  static_assert(t3<C>() == 2);
  // expected-error@-1 {{call to 't3' is ambiguous}}
  // expected-note@#canon-t3-0 {{candidate function}}
  // expected-note@#canon-t3-2 {{candidate function}}

  struct D { enum X {}; };
  static_assert(t3<D>() == 3);
  // expected-error@-1 {{call to 't3' is ambiguous}}
  // expected-note@#canon-t3-0 {{candidate function}}
  // expected-note@#canon-t3-3 {{candidate function}}

  template <class T> constexpr int t4(typename T::template X<int>* = 0) { return 0; }
  // expected-note@-1 2{{candidate function}}
  template <class T> constexpr int t4(struct   T::template X<int>* = 0) { return 1; }
  // expected-note@-1 2{{candidate function}}
  template <class T> constexpr int t4(union    T::template X<int>* = 0) { return 2; }
  // expected-note@-1 2{{candidate function}}

  struct E { template <class T> using X = T; };
  static_assert(t4<E>() == 0);

  struct F { template <class> struct X {}; };
  static_assert(t4<F>() == 1); // expected-error {{call to 't4' is ambiguous}}

  struct G { template <class> union X {}; };
  static_assert(t4<G>() == 2); // expected-error {{call to 't4' is ambiguous}}
} // namespace canon
