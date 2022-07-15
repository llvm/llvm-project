// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2a -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-error@+1 {{variadic macro}}
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
#endif

#if __cplusplus >= 201103L
namespace std {
  typedef decltype(sizeof(int)) size_t;

  template<typename E> class initializer_list {
    const E *begin;
    size_t size;

  public:
    initializer_list();
  };
} // std
#endif

namespace dr1601 { // dr1601: 10
enum E : char { e };
#if __cplusplus < 201103L
    // expected-error@-2 {{enumeration types with a fixed underlying type are a C++11 extension}}
#endif
void f(char);
void f(int);
void g() {
  f(e);
}
} // namespace dr1601

namespace dr1611 { // dr1611: dup 1658
  struct A { A(int); };
  struct B : virtual A { virtual void f() = 0; };
  struct C : B { C() : A(0) {} void f(); };
  C c;
}

namespace dr1684 { // dr1684: 3.6
#if __cplusplus >= 201103L
  struct NonLiteral { // expected-note {{because}}
    NonLiteral();
    constexpr int f() { return 0; } // expected-warning 0-1{{will not be implicitly 'const'}}
  };
  constexpr int f(NonLiteral &) { return 0; }
  constexpr int f(NonLiteral) { return 0; } // expected-error {{not a literal type}}
#endif
}

namespace dr1631 {  // dr1631: 3.7
#if __cplusplus >= 201103L
  // Incorrect overload resolution for single-element initializer-list

  struct A { int a[1]; };
  struct B { B(int); };
  void f(B, int);
  void f(B, int, int = 0);
  void f(int, A);

  void test() {
    f({0}, {{1}}); // expected-warning {{braces around scalar init}}
  }

  namespace with_error {
    void f(B, int);           // TODO: expected- note {{candidate function}}
    void f(int, A);           // expected-note {{candidate function}}
    void f(int, A, int = 0);  // expected-note {{candidate function}}
    
    void test() {
      f({0}, {{1}});        // expected-error{{call to 'f' is ambiguous}}
    }
  }
#endif
}

namespace dr1638 { // dr1638: yes
#if __cplusplus >= 201103L
  template<typename T> struct A {
    enum class E; // expected-note {{previous}}
    enum class F : T; // expected-note 2{{previous}}
  };

  template<> enum class A<int>::E;
  template<> enum class A<int>::E {};
  template<> enum class A<int>::F : int;
  template<> enum class A<int>::F : int {};

  template<> enum class A<short>::E : int;
  template<> enum class A<short>::E : int {};

  template<> enum class A<short>::F; // expected-error {{different underlying type}}
  template<> enum class A<char>::E : char; // expected-error {{different underlying type}}
  template<> enum class A<char>::F : int; // expected-error {{different underlying type}}

  enum class A<unsigned>::E; // expected-error {{template specialization requires 'template<>'}} expected-error {{nested name specifier}}
  template enum class A<unsigned>::E; // expected-error {{enumerations cannot be explicitly instantiated}}
  enum class A<unsigned>::E *e; // expected-error {{must use 'enum' not 'enum class'}}

  struct B {
    friend enum class A<unsigned>::E; // expected-error {{must use 'enum' not 'enum class'}}
  };
#endif
}

namespace dr1645 { // dr1645: 3.9
#if __cplusplus >= 201103L
  struct A {
    constexpr A(int, float = 0); // expected-note {{candidate}}
    explicit A(int, int = 0); // expected-note 2{{candidate}}
    A(int, int, int = 0) = delete; // expected-note {{candidate}}
  };

  struct B : A {
    using A::A; // expected-note 4{{inherited here}}
  };

  constexpr B a(0); // expected-error {{ambiguous}}
  constexpr B b(0, 0); // expected-error {{ambiguous}}
#endif
}

namespace dr1653 { // dr1653: 4 c++17
  void f(bool b) {
    ++b;
    b++;
#if __cplusplus <= 201402L
    // expected-warning@-3 {{deprecated}} expected-warning@-2 {{deprecated}}
#else
    // expected-error@-5 {{incrementing expression of type bool}} expected-error@-4 {{incrementing expression of type bool}}
#endif
    --b; // expected-error {{cannot decrement expression of type bool}}
    b--; // expected-error {{cannot decrement expression of type bool}}
    b += 1; // ok
    b -= 1; // ok
  }
}

namespace dr1658 { // dr1658: 5
  namespace DefCtor {
    class A { A(); }; // expected-note 0-2{{here}}
    class B { ~B(); }; // expected-note 0-2{{here}}

    // The stars align! An abstract class does not construct its virtual bases.
    struct C : virtual A { C(); virtual void foo() = 0; };
    C::C() = default; // ok, not deleted, expected-error 0-1{{extension}}
    struct D : virtual B { D(); virtual void foo() = 0; };
    D::D() = default; // ok, not deleted, expected-error 0-1{{extension}}

    // In all other cases, we are not so lucky.
    struct E : A { E(); virtual void foo() = 0; };
#if __cplusplus < 201103L
    E::E() = default; // expected-error {{private default constructor}} expected-error {{extension}} expected-note {{here}}
#else
    E::E() = default; // expected-error {{would delete}} expected-note@-4{{inaccessible default constructor}}
#endif
    struct F : virtual A { F(); };
#if __cplusplus < 201103L
    F::F() = default; // expected-error {{private default constructor}} expected-error {{extension}} expected-note {{here}}
#else
    F::F() = default; // expected-error {{would delete}} expected-note@-4{{inaccessible default constructor}}
#endif

    struct G : B { G(); virtual void foo() = 0; };
#if __cplusplus < 201103L
    G::G() = default; // expected-error@-2 {{private destructor}} expected-error {{extension}} expected-note {{here}}
#else
    G::G() = default; // expected-error {{would delete}} expected-note@-4{{inaccessible destructor}}
#endif
    struct H : virtual B { H(); };
#if __cplusplus < 201103L
    H::H() = default; // expected-error@-2 {{private destructor}} expected-error {{extension}} expected-note {{here}}
#else
    H::H() = default; // expected-error {{would delete}} expected-note@-4{{inaccessible destructor}}
#endif
  }

  namespace Dtor {
    class B { ~B(); }; // expected-note 0-2{{here}}

    struct D : virtual B { ~D(); virtual void foo() = 0; };
    D::~D() = default; // ok, not deleted, expected-error 0-1{{extension}}

    struct G : B { ~G(); virtual void foo() = 0; };
#if __cplusplus < 201103L
    G::~G() = default; // expected-error@-2 {{private destructor}} expected-error {{extension}} expected-note {{here}}
#else
    G::~G() = default; // expected-error {{would delete}} expected-note@-4{{inaccessible destructor}}
#endif
    struct H : virtual B { ~H(); };
#if __cplusplus < 201103L
    H::~H() = default; // expected-error@-2 {{private destructor}} expected-error {{extension}} expected-note {{here}}
#else
    H::~H() = default; // expected-error {{would delete}} expected-note@-4{{inaccessible destructor}}
#endif
  }

  namespace MemInit {
    struct A { A(int); }; // expected-note {{here}}
    struct B : virtual A {
      B() {}
      virtual void f() = 0;
    };
    struct C : virtual A {
      C() {} // expected-error {{must explicitly initialize}}
    };
  }

  namespace CopyCtorParamType {
    struct A { A(A&); };
    struct B : virtual A { virtual void f() = 0; };
    struct C : virtual A { virtual void f(); };
    struct D : A { virtual void f() = 0; };

    struct X {
      friend B::B(const B&) throw();
      friend C::C(C&);
      friend D::D(D&);
    };
  }

  namespace CopyCtor {
    class A { A(const A&); A(A&&); }; // expected-note 0-4{{here}} expected-error 0-1{{extension}}

    struct C : virtual A { C(const C&); C(C&&); virtual void foo() = 0; }; // expected-error 0-1{{extension}}
    C::C(const C&) = default; // expected-error 0-1{{extension}}
    C::C(C&&) = default; // expected-error 0-2{{extension}}

    struct E : A { E(const E&); E(E&&); virtual void foo() = 0; }; // expected-error 0-1{{extension}}
#if __cplusplus < 201103L
    E::E(const E&) = default; // expected-error {{private copy constructor}} expected-error {{extension}} expected-note {{here}}
    E::E(E&&) = default; // expected-error {{private move constructor}} expected-error 2{{extension}} expected-note {{here}}
#else
    E::E(const E&) = default; // expected-error {{would delete}} expected-note@-5{{inaccessible copy constructor}}
    E::E(E&&) = default; // expected-error {{would delete}} expected-note@-6{{inaccessible move constructor}}
#endif
    struct F : virtual A { F(const F&); F(F&&); }; // expected-error 0-1{{extension}}
#if __cplusplus < 201103L
    F::F(const F&) = default; // expected-error {{private copy constructor}} expected-error {{extension}} expected-note {{here}}
    F::F(F&&) = default; // expected-error {{private move constructor}} expected-error 2{{extension}} expected-note {{here}}
#else
    F::F(const F&) = default; // expected-error {{would delete}} expected-note@-5{{inaccessible copy constructor}}
    F::F(F&&) = default; // expected-error {{would delete}} expected-note@-6{{inaccessible move constructor}}
#endif
  }

  // assignment case is superseded by dr2180
}

namespace dr1672 { // dr1672: 7
  struct Empty {};
  struct A : Empty {};
  struct B { Empty e; };
  struct C : A { B b; int n; };
  struct D : A { int n; B b; };

  static_assert(!__is_standard_layout(C), "");
  static_assert(__is_standard_layout(D), "");

  struct E { B b; int n; };
  struct F { int n; B b; };
  union G { B b; int n; };
  union H { int n; B b; };

  struct X {};
  template<typename T> struct Y : X, A { T t; };

  static_assert(!__is_standard_layout(Y<E>), "");
  static_assert(__is_standard_layout(Y<F>), "");
  static_assert(!__is_standard_layout(Y<G>), "");
  static_assert(!__is_standard_layout(Y<H>), "");
  static_assert(!__is_standard_layout(Y<X>), "");
}

namespace dr1687 { // dr1687: 7
  template<typename T> struct To {
    operator T(); // expected-note 2{{first operand was implicitly converted to type 'int *'}}
    // expected-note@-1 {{second operand was implicitly converted to type 'double'}}
#if __cplusplus > 201703L
    // expected-note@-3 2{{operand was implicitly converted to type 'dr1687::E}}
#endif
  };

  int *a = To<int*>() + 100.0; // expected-error {{invalid operands to binary expression ('To<int *>' and 'double')}}
  int *b = To<int*>() + To<double>(); // expected-error {{invalid operands to binary expression ('To<int *>' and 'To<double>')}}

#if __cplusplus > 201703L
  enum E1 {};
  enum E2 {};
  auto c = To<E1>() <=> To<E2>(); // expected-error {{invalid operands to binary expression ('To<dr1687::E1>' and 'To<dr1687::E2>')}}
#endif
}

namespace dr1690 { // dr1690: 9
  // See also the various tests in "CXX/basic/basic.lookup/basic.lookup.argdep".
#if __cplusplus >= 201103L
  namespace N {
    static auto lambda = []() { struct S {} s; return s; };
    void f(decltype(lambda()));
  }

  void test() {
    auto s = N::lambda();
    f(s); // ok
  }
#endif
}

namespace dr1691 { // dr1691: 9
#if __cplusplus >= 201103L
  namespace N {
    namespace M {
      enum E : int;
      void f(E);
    }
    enum M::E : int {};
    void g(M::E); // expected-note {{declared here}}
  }
  void test() {
    N::M::E e;
    f(e); // ok
    g(e); // expected-error {{use of undeclared}}
  }
#endif
}

namespace dr1692 { // dr1692: 9
  namespace N {
    struct A {
      struct B {
        struct C {};
      };
    };
    void f(A::B::C);
  }
  void test() {
    N::A::B::C c;
    f(c); // ok
  }
}

namespace dr1696 { // dr1696: 7
  namespace std_examples {
#if __cplusplus >= 201402L
    extern struct A a;
    struct A {
      const A &x = { A{a, a} };
      const A &y = { A{} }; // expected-error {{default member initializer for 'y' needed within definition of enclosing class 'A' outside of member functions}} expected-note {{here}}
    };
    A a{a, a};
#endif
  }

  struct A { A(); ~A(); };
#if __cplusplus >= 201103L
  struct B {
    A &&a; // expected-note {{declared here}}
    B() : a{} {} // expected-error {{reference member 'a' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
  } b;
#endif

  struct C {
    C();
    const A &a; // expected-note {{declared here}}
  };
  C::C() : a(A()) {} // expected-error {{reference member 'a' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}

#if __cplusplus >= 201103L
  // This is OK in C++14 onwards, per DR1815, though we don't support that yet:
  //   D1 d1 = {};
  // is equivalent to
  //   D1 d1 = {A()};
  // ... which lifetime-extends the A temporary.
  struct D1 {
#if __cplusplus < 201402L
    // expected-error@-2 {{binds to a temporary}}
#endif
    const A &a = A(); // expected-note {{default member init}}
  };
  D1 d1 = {};
#if __cplusplus < 201402L
    // expected-note@-2 {{first required here}}
#else
    // expected-warning-re@-4 {{sorry, lifetime extension {{.*}} not supported}}
#endif

  struct D2 {
    const A &a = A(); // expected-note {{default member init}}
    D2() {} // expected-error {{binds to a temporary}}
  };

  struct D3 { // expected-error {{binds to a temporary}}
    const A &a = A(); // expected-note {{default member init}}
  };
  D3 d3; // expected-note {{first required here}}

  struct haslist1 {
    std::initializer_list<int> il; // expected-note {{'std::initializer_list' member}}
    haslist1(int i) : il{i, 2, 3} {} // expected-error {{backing array for 'std::initializer_list' member 'il' is a temporary object}}
  };

  struct haslist2 {
    std::initializer_list<int> il; // expected-note {{'std::initializer_list' member}}
    haslist2();
  };
  haslist2::haslist2() : il{1, 2} {} // expected-error {{backing array for 'std::initializer_list' member 'il' is a temporary object}}

  struct haslist3 {
    std::initializer_list<int> il = {1, 2, 3};
  };

  struct haslist4 { // expected-error {{backing array for 'std::initializer_list' member 'il' is a temporary object}}
    std::initializer_list<int> il = {1, 2, 3}; // expected-note {{default member initializer}}
  };
  haslist4 hl4; // expected-note {{in implicit default constructor}}

  struct haslist5 {
    std::initializer_list<int> il = {1, 2, 3}; // expected-note {{default member initializer}}
    haslist5() {} // expected-error {{backing array for 'std::initializer_list' member 'il' is a temporary object}}
  };
#endif
}
