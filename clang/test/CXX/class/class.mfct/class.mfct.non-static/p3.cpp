// RUN: %clang_cc1 -fsyntax-only -verify %s

// [class.mfct.non-static]p3:
//   When an id-expression (5.1) that is not part of a class member
//   access syntax (5.2.5) and not used to form a pointer to member
//   (5.3.1) is used in the body of a non-static member function of
//   class X, if name lookup (3.4.1) resolves the name in the
//   id-expression to a non-static non-type member of some class C,
//   the id-expression is transformed into a class member access
//   expression (5.2.5) using (*this) (9.3.2) as the
//   postfix-expression to the left of the . operator. [ Note: if C is
//   not X or a base class of X, the class member access expression is
//   ill-formed. --end note] Similarly during name lookup, when an
//   unqualified-id (5.1) used in the definition of a member function
//   for class X resolves to a static member, an enumerator or a
//   nested type of class X or of a base class of X, the
//   unqualified-id is transformed into a qualified-id (5.1) in which
//   the nested-name-specifier names the class of the member function.

namespace test0 {
  class A {
    int data_member;
    int instance_method();
    static int static_method();

    bool test() {
      return data_member + instance_method() < static_method();
    }
  };
}

namespace test1 {
  struct Opaque1 {}; struct Opaque2 {}; struct Opaque3 {};

  struct A {
    void foo(Opaque1); // expected-note {{candidate}}
    void foo(Opaque2); // expected-note {{candidate}}
  };

  struct B : A {
    void test();
  };

  struct C1 : A { };
  struct C2 : B { };

  void B::test() {
    A::foo(Opaque1());
    A::foo(Opaque2());
    A::foo(Opaque3()); // expected-error {{no matching member function}}

    C1::foo(Opaque1()); // expected-error {{call to non-static member function without an object argument}}
    C2::foo(Opaque1()); // expected-error {{call to non-static member function without an object argument}}
  }
}

namespace test2 {
  struct Unrelated {
    void foo();
  };

  template <class T> struct B;
  template <class T> struct C;

  template <class T> struct A {
    void foo();

    void test0() {
      Unrelated::foo(); // expected-error {{call to non-static member function without an object argument}}
    }

    void test1() {
      B<T>::foo(); // expected-error {{call to non-static member function without an object argument}}
    }

    static void test2() {
      B<T>::foo(); // expected-error {{call to non-static member function without an object argument}}
    }

    void test3() {
      C<T>::foo(); // expected-error {{no member named 'foo'}}
    }
  };

  template <class T> struct B : A<T> {
  };

  template <class T> struct C {
  };

  int test() {
    A<int> a;
    a.test0(); // no instantiation note here, decl is ill-formed
    a.test1(); // expected-note {{in instantiation}}
    a.test2(); // expected-note {{in instantiation}}
    a.test3(); // expected-note {{in instantiation}}
  }
}

namespace test3 {
  struct A {
    void f0();

    template<typename T>
    void f1();

    static void f2();

    template<typename T>
    static void f3();

    int x0;

    static constexpr int x1 = 0;

    template<typename T>
    static constexpr int x2 = 0;
  };

  template<typename T>
  struct B : T {
    auto g0() -> decltype(T::f0());

    auto g1() -> decltype(T::template f1<int>());

    auto g2() -> decltype(T::f2());

    auto g3() -> decltype(T::template f3<int>());

    auto g4() -> decltype(T::x0);

    auto g5() -> decltype(T::x1);

    auto g6() -> decltype(T::template x2<int>);

    decltype(T::f0()) g7(); // expected-error {{call to non-static member function without an object argument}}

    decltype(T::template f1<int>()) g8(); // expected-error {{call to non-static member function without an object argument}}

    decltype(T::f2()) g9();

    decltype(T::template f3<int>()) g10();

    decltype(T::x0) g11();

    decltype(T::x1) g12();

    decltype(T::template x2<int>) g13();
  };

  template struct B<A>; // expected-note {{in instantiation of}}

  template<typename T>
  struct C : T {
    static auto g0() -> decltype(T::f0()); // expected-error {{'this' cannot be implicitly used in a static member function declaration}}

    static auto g1() -> decltype(T::template f1<int>()); // expected-error {{'this' cannot be implicitly used in a static member function declaration}}

    static auto g2() -> decltype(T::f2());

    static auto g3() -> decltype(T::template f3<int>());

    static auto g4() -> decltype(T::x0); // expected-error {{'this' cannot be implicitly used in a static member function declaration}}

    static auto g5() -> decltype(T::x1);

    static auto g6() -> decltype(T::template x2<int>);

    static decltype(T::f0()) g7(); // expected-error {{call to non-static member function without an object argument}}

    static decltype(T::template f1<int>()) g8(); // expected-error {{call to non-static member function without an object argument}}

    static decltype(T::f2()) g9();

    static decltype(T::template f3<int>()) g10();

    static decltype(T::x0) g11();

    static decltype(T::x1) g12();

    static decltype(T::template x2<int>) g13();
  };

  template struct C<A>; // expected-note {{in instantiation of}}
}
