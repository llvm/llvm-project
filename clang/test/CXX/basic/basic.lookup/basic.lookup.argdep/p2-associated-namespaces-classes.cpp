// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s

// Attempt to test each rule for forming associated namespaces
// and classes as described in [basic.lookup.argdep]p2.

// fundamental type: no associated namespace and no associated class
namespace adl_fundamental_type {
  constexpr int g(char) { return 1; } // #1
  template <typename T> constexpr int foo(T t) { return g(t); }
  constexpr int g(int) { return 2; } // #2 not found
  void test() {
    static_assert(foo(0) == 1); // ok, #1
  }
}

// class type:
//   associated classes: itself, the class of which it is a member (if any),
//                       direct and indirect base classes
//   associated namespaces: innermost enclosing namespaces of associated classes
namespace adl_class_type {
  // associated class: itself, simple case
  namespace X1 {
    namespace N {
      struct S {};
      void f(S); // found
    }
    void g(N::S); // not found
  };
  void test1() {
    f(X1::N::S{}); // ok
    g(X1::N::S{}); // expected-error {{use of undeclared identifier}}
  }

  // associated class: itself, local type
  namespace X2 {
    auto foo() {
      struct S {} s;
      return s;
    }
    using S = decltype(foo());
    void f(S); // #1
  }
  void test2() {
    f(X2::S{}); // This is well-formed; X2 is the innermost enclosing namespace
                // of the local struct S. Calls #1.
  }

  // associated class: the parent class
  namespace X3 {
    struct S {
      struct T {};
      friend void f(T);
    };
  }
  void test3() {
    f(X3::S::T{}); // ok
  }

  // associated class: direct and indirect base classes
  namespace X4 {
    namespace IndirectBaseNamespace {
      struct IndirectBase {};
      void f(IndirectBase); // #1
    }
    namespace DirectBaseNamespace {
      struct DirectBase : IndirectBaseNamespace::IndirectBase {};
      void g(DirectBase); // #2
    }
    struct S : DirectBaseNamespace::DirectBase {};
  }
  void test4() {
    f(X4::S{}); // ok, #1
    g(X4::S{}); // ok, #2
  }

  // associated class: itself, lambda
  namespace X5 {
    namespace N {
      auto get_lambda() { return [](){}; }
      void f(decltype(get_lambda()));
    }

    void test5() {
      auto lambda = N::get_lambda();
      f(lambda); // ok
    }
  }

  // The parameter types and return type of a lambda's operator() do not
  // contribute to the associated namespaces and classes of the lambda itself.
  namespace X6 {
    namespace N {
      struct A {};
      template<class T> constexpr int f(T) { return 1; }
    }

    constexpr int f(N::A (*)()) { return 2; }
    constexpr int f(void (*)(N::A)) { return 3; }

    void test() {
      constexpr auto lambda = []() -> N::A { return {}; };
      static_assert(f(lambda) == 2);

      constexpr auto lambda2 = [](N::A) {};
      static_assert(f(lambda2) == 3);
    }
  }
} // namespace adl_class_type

// class template specialization: as for class type plus
//   for non-type template arguments:
//    - nothing
//   for type template arguments:
//    - associated namespaces and classes of the type template arguments
//   for template template arguments:
//    - namespaces of which template template arguments are member of
//    - classes of which member template used as template template arguments
//      are member of
namespace adl_class_template_specialization_type {
  // non-type template argument
  namespace X1 {
    namespace BaseNamespace { struct Base {}; }
    namespace N { struct S : BaseNamespace::Base {}; }
    template <N::S *> struct C {};
    namespace N {
      template <S *p> void X1_f(C<p>); // #1
    }
    namespace BaseNamespace {
      template <N::S *p> void X1_g(C<p>); // #2
    }
    template <N::S *p> void X1_h(C<p>); // #3
  }
  void test1() {
    constexpr X1::N::S *p = nullptr;
    X1::C<p> c;
    X1_f(c); // N is not added to the set of associated namespaces
             // and #1 is not found...
             // expected-error@-2 {{use of undeclared identifier}}
    X1_g(c); // ... nor is #2 ...
             // expected-error@-1 {{use of undeclared identifier}}
    X1_h(c); // ... but the namespace X1 is added and #3 is found.
  }

  // type template argument
  namespace X2 {
    template <typename T> struct C {};
    namespace BaseNamespace { struct Base {}; }
    namespace N { struct S : BaseNamespace::Base {}; }
    namespace N {
      template <typename T> void X2_f(C<T>); // #1
    }
    namespace BaseNamespace {
      template <typename T> void X2_g(C<T>); // #2
    }
    template <typename T> void X2_h(C<T>); // #2
  }
  void test2() {
    X2::C<X2::N::S> c;
    X2_f(c); // N is added to the set of associated namespaces and #1 is found.
    X2_g(c); // Similarly BaseNamespace is added and #2 is found.
    X2_h(c); // As before, X2 is also added and #3 is found.
  }

  // template template argument
  namespace X3 {
    template <template <typename> class TT> struct C {};
    namespace N {
      template <typename T> struct Z {};
      void X3_f(C<Z>); // #1
    }
    struct M {
      template <typename T> struct Z {};
      friend void X3_g(C<Z>); // #2
    };
  }
  void test3() {
    X3::C<X3::N::Z> c1;
    X3::C<X3::M::Z> c2;
    X3_f(c1); // ok, namespace N is added, #1
    X3_g(c2); // ok, struct M is added, #2
  }
}

// enumeration type:
//  associated namespace: innermost enclosing namespace of its declaration.
//  associated class: if the enumeration is a class member, the member's class.
namespace adl_enumeration_type {
  namespace N {
    enum E : int;
    void f(E);
    struct S {
      enum F : int;
      friend void g(F);
    };
    auto foo() {
      enum G {} g;
      return g;
    }
    using G = decltype(foo());
    void h(G);
  }

  void test() {
    N::E e;
    f(e); // ok
    N::S::F f;
    g(f); // ok
    N::G g;
    h(g); // ok

  }
}

// pointer and reference type:
//  associated namespaces and classes of the pointee type
// array type:
//  associated namespaces and classes of the base type
namespace adl_point_array_reference_type {
  namespace N {
    struct S {};
    void f(S *);
    void f(S &);
  }

  void test() {
    N::S *p;
    f(p); // ok
    extern N::S &r;
    f(r); // ok
    N::S a[2];
    f(a); // ok
  }
}

// function type:
//  associated namespaces and classes of the function parameter types
//  and the return type.
namespace adl_function_type {
  namespace M { struct T; }
  namespace N {
    struct S {};
    void f(S (*)(M::T));
  };
  namespace M {
    struct T {};
    void g(N::S (*)(T));
  }

  void test() {
    extern N::S x(M::T);
    f(x); // ok
    g(x); // ok
  }
}

// pointer to member function:
//  associated namespaces and classes of the class, parameter types
//  and return type.
namespace adl_pointer_to_member_function {
  namespace M { struct C; }
  namespace L { struct T; }
  namespace N {
    struct S {};
    void f(N::S (M::C::*)(L::T));
  }
  namespace L {
    struct T {};
    void g(N::S (M::C::*)(L::T));
  }
  namespace M {
    struct C {};
    void h(N::S (M::C::*)(L::T));
  }

  void test() {
    N::S (M::C::*p)(L::T);
    f(p); // ok
    g(p); // ok
    h(p); // ok
  }
}

// pointer to member:
//  associated namespaces and classes of the class and of the member type.
namespace adl_pointer_to_member {
  namespace M { struct C; }
  namespace N {
    struct S {};
    void f(N::S (M::C::*));
  }
  namespace M {
    struct C {};
    void g(N::S (M::C::*));
  }

  void test() {
    N::S (M::C::*p);
    f(p); // ok
    g(p); // ok
  }
}

// [...] if the argument is the name or address of a set of overloaded
// functions and/or function templates, its associated classes and namespaces
// are the union of those associated with each of the members of the set,
// i.e., the classes and namespaces associated with its parameter types and
// return type.
//
// Additionally, if the aforementioned set of overloaded functions is named
// with a template-id, its associated classes and namespaces also include
// those of its type template-arguments and its template template-arguments.
//
// CWG 33 for the union rule. CWG 997 for the template-id rule.
namespace adl_overload_set {
  namespace N {
    struct S {};
    constexpr int f(int (*g)()) { return g(); }
    // expected-note@-1 2{{'N::f' declared here}}
    template <typename T> struct Q;
  }

  constexpr int g1() { return 1; }
  constexpr int g1(N::S) { return 2; }

  template <typename T> constexpr int g2() { return 3; }

  // Inspired from CWG 997.
  constexpr int g3() { return 4; }
  template <typename T> constexpr int g3(T, N::Q<T>) { return 5; }

  void test() {
    static_assert(f(g1) == 1, "");        // Well-formed from the union rule above
    static_assert(f(g2<N::S>) == 3, "");  // FIXME: Well-formed from the template-id rule above.
                                          // expected-error@-1 {{use of undeclared}}

    // A objection was raised during review against implementing the
    // template-id rule. Currently only GCC implements it. Implementing
    // it would weaken the argument to remove it in the future since
    // actual real code might start to depend on it.

    static_assert(f(g3) == 4, "");        // FIXME: Also well-formed from the union rule.
                                          // expected-error@-1 {{use of undeclared}}
  }
}
