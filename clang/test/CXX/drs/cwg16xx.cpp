// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected,cxx98-14,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,cxx98-14,since-cxx11,cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,since-cxx14,cxx98-14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,since-cxx14,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,since-cxx14,since-cxx20,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx23,since-cxx14,since-cxx20,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx23,since-cxx14,since-cxx20,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
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

namespace cwg1601 { // cwg1601: 10
enum E : char { e };
// cxx98-error@-1 {{enumeration types with a fixed underlying type are a C++11 extension}}
void f(char);
void f(int);
void g() {
  f(e);
}
} // namespace cwg1601

namespace cwg1606 { // cwg1606: 3.1
#if __cplusplus >= 201103L
  std::size_t test() {
    int i = 1;
    int j = 1;
    auto f = [=]{ return i + j; };
    return sizeof(f);
  }
#endif
} // namespace cwg1606

namespace cwg1611 { // cwg1611: dup 1658
  struct A { A(int); };
  struct B : virtual A { virtual void f() = 0; };
  struct C : B { C() : A(0) {} void f(); };
  C c;
}

namespace cwg1631 {  // cwg1631: 3.7
#if __cplusplus >= 201103L
  // Incorrect overload resolution for single-element initializer-list

  struct A { int a[1]; };
  struct B { B(int); };
  void f(B, int);
  void f(B, int, int = 0);
  void f(int, A);

  void test() {
    f({0}, {{1}});
    // since-cxx11-warning@-1 {{braces around scalar initializer}}
  }

  namespace with_error {
    void f(B, int);           // TODO: expected- note {{candidate function}}
    void f(int, A);           // #cwg1631-f
    void f(int, A, int = 0);  // #cwg1631-f-int

    void test() {
      f({0}, {{1}});
      // since-cxx11-error@-1 {{call to 'f' is ambiguous}}
      //   since-cxx11-note@#cwg1631-f {{candidate function}}
      //   since-cxx11-note@#cwg1631-f-int {{candidate function}}
    }
  }
#endif
}

namespace cwg1638 { // cwg1638: 3.1
#if __cplusplus >= 201103L
  template<typename T> struct A {
    enum class E; // #cwg1638-E
    enum class F : T; // #cwg1638-F
  };

  template<> enum class A<int>::E;
  template<> enum class A<int>::E {};
  template<> enum class A<int>::F : int;
  template<> enum class A<int>::F : int {};

  template<> enum class A<short>::E : int;
  template<> enum class A<short>::E : int {};

  template<> enum class A<short>::F;
  // since-cxx11-error@-1 {{enumeration redeclared with different underlying type 'int' (was 'short')}}
  //   since-cxx11-note@#cwg1638-F {{previous declaration is here}}
  template<> enum class A<char>::E : char;
  // since-cxx11-error@-1 {{enumeration redeclared with different underlying type 'char' (was 'int')}}
  //   since-cxx11-note@#cwg1638-E {{previous declaration is here}}
  template<> enum class A<char>::F : int;
  // since-cxx11-error@-1 {{enumeration redeclared with different underlying type 'int' (was 'char')}}
  //   since-cxx11-note@#cwg1638-F {{previous declaration is here}}

  enum class A<unsigned>::E;
  // since-cxx11-error@-1 {{template specialization requires 'template<>'}}
  template enum class A<unsigned>::E;
  // since-cxx11-error@-1 {{enumerations cannot be explicitly instantiated}}
  enum class A<unsigned>::E *e;
  // since-cxx11-error@-1 {{reference to enumeration must use 'enum' not 'enum class'}}

  struct B {
    friend enum class A<unsigned>::E;
    // since-cxx11-error@-1 {{reference to enumeration must use 'enum' not 'enum class'}}
    // since-cxx11-error@-2 {{elaborated enum specifier cannot be declared as a friend}}
    // since-cxx11-note@-3 {{remove 'enum class' to befriend an enum}}
  };
#endif
}

namespace cwg1645 { // cwg1645: 3.9
#if __cplusplus >= 201103L
  struct A {
    constexpr A(int, float = 0); // #cwg1645-int-float
    explicit A(int, int = 0); // #cwg1645-int-int
    A(int, int, int = 0) = delete; // #cwg1645-int-int-int
  };

  struct B : A {
    using A::A; // #cwg1645-using
  };

  constexpr B a(0);
  // since-cxx11-error@-1 {{call to constructor of 'const B' is ambiguous}}
  //   since-cxx11-note@#cwg1645-int-float {{candidate inherited constructor}}
  //   since-cxx11-note@#cwg1645-using {{constructor from base class 'A' inherited here}}
  //   since-cxx11-note@#cwg1645-int-int {{candidate inherited constructor}}
  //   since-cxx11-note@#cwg1645-using {{constructor from base class 'A' inherited here}}
  constexpr B b(0, 0);
  // since-cxx11-error@-1 {{call to constructor of 'const B' is ambiguous}}
  //   since-cxx11-note@#cwg1645-int-int {{candidate inherited constructor}}
  //   since-cxx11-note@#cwg1645-using {{constructor from base class 'A' inherited here}}
  //   since-cxx11-note@#cwg1645-int-int-int {{candidate inherited constructor has been explicitly deleted}}
  //   since-cxx11-note@#cwg1645-using {{constructor from base class 'A' inherited here}}
#endif
}

namespace cwg1652 { // cwg1652: 3.6
  int a, b;
  static_assert(&a + 1 == &b, "");
  // expected-error@-1 {{static assertion expression is not an integral constant expression}}
  //   expected-note@-2 {{comparison against pointer '&a + 1' that points past the end of a complete object has unspecified value}}
}

namespace cwg1653 { // cwg1653: 4 c++17
  void f(bool b) {
    ++b;
    // cxx98-14-warning@-1 {{incrementing expression of type bool is deprecated and incompatible with C++17}}
    // since-cxx17-error@-2 {{SO C++17 does not allow incrementing expression of type bool}}
    b++;
    // cxx98-14-warning@-1 {{incrementing expression of type bool is deprecated and incompatible with C++17}}
    // since-cxx17-error@-2 {{SO C++17 does not allow incrementing expression of type bool}}
    --b;
    // expected-error@-1 {{cannot decrement expression of type bool}}
    b--;
    // expected-error@-1 {{cannot decrement expression of type bool}}
    b += 1; // ok
    b -= 1; // ok
  }
}

namespace cwg1658 { // cwg1658: 5
  namespace DefCtor {
    class A { A(); }; // #cwg1658-A1
    class B { ~B(); }; // #cwg1658-B1

    // The stars align! An abstract class does not construct its virtual bases.
    struct C : virtual A { C(); virtual void foo() = 0; };
    C::C() = default; // ok, not deleted
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
    struct D : virtual B { D(); virtual void foo() = 0; };
    D::D() = default; // ok, not deleted
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}

    // In all other cases, we are not so lucky.
    struct E : A { E(); virtual void foo() = 0; }; // #cwg1658-E1
    E::E() = default; // #cwg1658-E1-ctor
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
    // cxx98-error@-2 {{base class 'A' has private default constructor}}
    //   cxx98-note@-3 {{in defaulted default constructor for 'cwg1658::DefCtor::E' first required here}}
    //   cxx98-note@#cwg1658-A1 {{implicitly declared private here}}
    // since-cxx11-error@#cwg1658-E1-ctor {{defaulting this default constructor would delete it after its first declaration}}
    //   since-cxx11-note@#cwg1658-E1 {{default constructor of 'E' is implicitly deleted because base class 'A' has an inaccessible default constructor}}
    struct F : virtual A { F(); }; // #cwg1658-F1
    F::F() = default; // #cwg1658-F1-ctor
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
    // cxx98-error@-2 {{inherited virtual base class 'A' has private default constructor}}
    //   cxx98-note@-3 {{in defaulted default constructor for 'cwg1658::DefCtor::F' first required here}}
    //   cxx98-note@#cwg1658-A1 {{implicitly declared private here}}
    // since-cxx11-error@#cwg1658-F1-ctor {{defaulting this default constructor would delete it after its first declaration}}
    //   since-cxx11-note@#cwg1658-F1 {{default constructor of 'F' is implicitly deleted because base class 'A' has an inaccessible default constructor}}

    struct G : B { G(); virtual void foo() = 0; }; // #cwg1658-G1
    G::G() = default; // #cwg1658-G1-ctor
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
    // cxx98-error@#cwg1658-G1 {{base class 'B' has private destructor}}
    //   cxx98-note@#cwg1658-G1-ctor {{in defaulted default constructor for 'cwg1658::DefCtor::G' first required here}}
    //   cxx98-note@#cwg1658-B1 {{implicitly declared private here}}
    // since-cxx11-error@#cwg1658-G1-ctor {{defaulting this default constructor would delete it after its first declaration}}
    //   since-cxx11-note@#cwg1658-G1 {{default constructor of 'G' is implicitly deleted because base class 'B' has an inaccessible destructor}}
    struct H : virtual B { H(); }; // #cwg1658-H1
    H::H() = default; // #cwg1658-H1-ctor
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
    // cxx98-error@#cwg1658-H1 {{base class 'B' has private destructor}}
    //   cxx98-note@#cwg1658-H1-ctor {{in defaulted default constructor for 'cwg1658::DefCtor::H' first required here}}
    //   cxx98-note@#cwg1658-B1 {{implicitly declared private here}}
    // since-cxx11-error@#cwg1658-H1-ctor {{defaulting this default constructor would delete it after its first declaration}}
    //   since-cxx11-note@#cwg1658-H1 {{default constructor of 'H' is implicitly deleted because base class 'B' has an inaccessible destructor}}
  }

  namespace Dtor {
    class B { ~B(); }; // #cwg1658-B2

    struct D : virtual B { ~D(); virtual void foo() = 0; };
    D::~D() = default; // ok, not deleted
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}

    struct G : B { ~G(); virtual void foo() = 0; }; // #cwg1658-G2
    G::~G() = default; // #cwg1658-G2-dtor
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
    // cxx98-error@#cwg1658-G2 {{base class 'B' has private destructor}}
    //   cxx98-note@#cwg1658-G2-dtor {{in defaulted destructor for 'cwg1658::Dtor::G' first required here}}
    //   cxx98-note@#cwg1658-B2 {{implicitly declared private here}}
    // since-cxx11-error@#cwg1658-G2-dtor {{defaulting this destructor would delete it after its first declaration}}
    //   since-cxx11-note@#cwg1658-G2 {{destructor of 'G' is implicitly deleted because base class 'B' has an inaccessible destructor}}
    struct H : virtual B { ~H(); }; // #cwg1658-H2
    H::~H() = default; // #cwg1658-H2-dtor
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
    // cxx98-error@#cwg1658-H2 {{base class 'B' has private destructor}}
    //   cxx98-note@#cwg1658-H2-dtor {{in defaulted destructor for 'cwg1658::Dtor::H' first required here}}
    //   cxx98-note@#cwg1658-B2 {{implicitly declared private here}}
    // since-cxx11-error@#cwg1658-H2-dtor {{defaulting this destructor would delete it after its first declaration}}
    //   since-cxx11-note@#cwg1658-H2 {{destructor of 'H' is implicitly deleted because base class 'B' has an inaccessible destructor}}
  }

  namespace MemInit {
    struct A { A(int); }; // #cwg1658-A3
    struct B : virtual A {
      B() {}
      virtual void f() = 0;
    };
    struct C : virtual A {
      C() {}
      // expected-error@-1 {{constructor for 'cwg1658::MemInit::C' must explicitly initialize the base class 'A' which does not have a default constructor}}
      //   expected-note@#cwg1658-A3 {{'cwg1658::MemInit::A' declared here}}
    };
  }

  namespace CopyCtorParamType {
    struct A { A(A&); };
    struct B : virtual A { virtual void f() = 0; };
    struct C : virtual A { virtual void f(); };
    struct D : A { virtual void f() = 0; }; // since-cxx23-note {{previous declaration is here}}

    struct X {
      friend B::B(const B&) throw();
      friend C::C(C&);
      friend D::D(D&); // since-cxx23-error {{non-constexpr declaration of 'D' follows constexpr declaration}}
    };
  }

  namespace CopyCtor {
    class A { A(const A&); A(A&&); }; // #cwg1658-A5
    // cxx98-error@-1 {{rvalue references are a C++11 extension}}

    struct C : virtual A { C(const C&); C(C&&); virtual void foo() = 0; };
    // cxx98-error@-1 {{rvalue references are a C++11 extension}}
    C::C(const C&) = default;
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
    C::C(C&&) = default;
    // cxx98-error@-1 {{rvalue references are a C++11 extension}}
    // cxx98-error@-2 {{defaulted function definitions are a C++11 extension}}

    struct E : A { E(const E&); E(E&&); virtual void foo() = 0; }; // #cwg1658-E5
    // cxx98-error@-1 {{rvalue references are a C++11 extension}}
    E::E(const E&) = default; // #cwg1658-E5-copy-ctor
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
    // cxx98-error@-2 {{base class 'A' has private copy constructor}}
    //   cxx98-note@-3 {{in defaulted copy constructor for 'cwg1658::CopyCtor::E' first required here}}
    //   cxx98-note@#cwg1658-A5 {{implicitly declared private here}}
    // since-cxx11-error@#cwg1658-E5-copy-ctor {{defaulting this copy constructor would delete it after its first declaration}}
    //   since-cxx11-note@#cwg1658-E5 {{copy constructor of 'E' is implicitly deleted because base class 'A' has an inaccessible copy constructor}}
    E::E(E&&) = default; // #cwg1658-E5-move-ctor
    // cxx98-error@-1 {{rvalue references are a C++11 extension}}
    // cxx98-error@-2 {{defaulted function definitions are a C++11 extension}}
    // cxx98-error@-3 {{base class 'A' has private move constructor}}
    //   cxx98-note@-4 {{in defaulted move constructor for 'cwg1658::CopyCtor::E' first required here}}
    //   cxx98-note@#cwg1658-A5 {{implicitly declared private here}}
    // since-cxx11-error@#cwg1658-E5-move-ctor {{defaulting this move constructor would delete it after its first declaration}}
    //   since-cxx11-note@#cwg1658-E5 {{move constructor of 'E' is implicitly deleted because base class 'A' has an inaccessible move constructor}}
    struct F : virtual A { F(const F&); F(F&&); }; // #cwg1658-F5
    // cxx98-error@-1 {{rvalue references are a C++11 extension}}
    F::F(const F&) = default; // #cwg1658-F5-copy-ctor
    // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
    // cxx98-error@-2 {{inherited virtual base class 'A' has private copy constructor}}
    //   cxx98-note@-3 {{in defaulted copy constructor for 'cwg1658::CopyCtor::F' first required here}}
    //   cxx98-note@#cwg1658-A5 {{implicitly declared private here}}
    // since-cxx11-error@#cwg1658-F5-copy-ctor {{defaulting this copy constructor would delete it after its first declaration}}
    //   since-cxx11-note@#cwg1658-F5 {{copy constructor of 'F' is implicitly deleted because base class 'A' has an inaccessible copy constructor}}
    F::F(F&&) = default; // #cwg1658-F5-move-ctor
    // cxx98-error@-1 {{rvalue references are a C++11 extension}}
    // cxx98-error@-2 {{defaulted function definitions are a C++11 extension}}
    // cxx98-error@-3 {{inherited virtual base class 'A' has private move constructor}}
    //   cxx98-note@-4 {{in defaulted move constructor for 'cwg1658::CopyCtor::F' first required here}}
    //   cxx98-note@#cwg1658-A5 {{implicitly declared private here}}
    // since-cxx11-error@#cwg1658-F5-move-ctor {{defaulting this move constructor would delete it after its first declaration}}
    //   since-cxx11-note@#cwg1658-F5 {{move constructor of 'F' is implicitly deleted because base class 'A' has an inaccessible move constructor}}
  }

  // assignment case is superseded by cwg2180
}

namespace cwg1672 { // cwg1672: 7
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

namespace cwg1684 { // cwg1684: 3.6
#if __cplusplus >= 201103L
  struct NonLiteral { // #cwg1684-struct
    NonLiteral();
    constexpr int f() { return 0; }
    // cxx11-warning@-1 {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const' to avoid a change in behavior}}
  };
  constexpr int f(NonLiteral &) { return 0; }
  constexpr int f(NonLiteral) { return 0; }
  // cxx11-20-error@-1 {{constexpr function's 1st parameter type 'NonLiteral' is not a literal type}}
  //   cxx11-20-note@#cwg1684-struct {{'NonLiteral' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
#endif
}

namespace cwg1687 { // cwg1687: 7
  template<typename T> struct To {
    operator T(); // #cwg1687-op-T
  };

  int *a = To<int*>() + 100.0;
  // expected-error@-1 {{invalid operands to binary expression ('To<int *>' and 'double')}}
  //   expected-note@#cwg1687-op-T {{first operand was implicitly converted to type 'int *'}}
  //   since-cxx20-note@#cwg1687-op-T {{second operand was implicitly converted to type 'cwg1687::E2'}}
  int *b = To<int*>() + To<double>();
  // expected-error@-1 {{invalid operands to binary expression ('To<int *>' and 'To<double>')}}
  //   expected-note@#cwg1687-op-T {{first operand was implicitly converted to type 'int *'}}
  //   expected-note@#cwg1687-op-T {{second operand was implicitly converted to type 'double'}}

#if __cplusplus >= 202002L
  enum E1 {};
  enum E2 {};
  auto c = To<E1>() <=> To<E2>();
  // since-cxx20-error@-1 {{invalid operands to binary expression ('To<E1>' and 'To<E2>')}}
  //   since-cxx20-note@#cwg1687-op-T {{operand was implicitly converted to type 'cwg1687::E}}
#endif
}

namespace cwg1690 { // cwg1690: 9
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

namespace cwg1691 { // cwg1691: 9
#if __cplusplus >= 201103L
  namespace N {
    namespace M {
      enum E : int;
      void f(E);
    }
    enum M::E : int {};
    void g(M::E); // #cwg1691-g
  }
  void test() {
    N::M::E e;
    f(e); // ok
    g(e);
    // since-cxx11-error@-1 {{use of undeclared identifier 'g'; did you mean 'N::g'?}}
    //   since-cxx11-note@#cwg1691-g {{'N::g' declared here}}
  }
#endif
}

namespace cwg1692 { // cwg1692: 9
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

namespace cwg1696 { // cwg1696: 7
  namespace std_examples {
#if __cplusplus >= 201402L
    extern struct A a;
    struct A {
      const A &x = { A{a, a} };
      const A &y = { A{} };
      // since-cxx14-error@-1 {{default member initializer for 'y' needed within definition of enclosing class 'A' outside of member functions}}
      //   since-cxx14-note@-2 {{default member initializer declared here}}
    };
    A a{a, a};

    struct A1 {
      A1() : v(42) {}
      // since-cxx14-error@-1 {{reference member 'v' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
      // since-cxx14-note@#cwg1696-A1 {{reference member declared here}}
      const int &v; // #cwg1696-A1
    };

    struct A2 {
      A2() = default;
      // since-cxx14-error@-1 {{reference member 'v' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
      // since-cxx14-note-re@#cwg1696-A2-b {{in defaulted default constructor for {{.*}} first required here}}
      // since-cxx14-note@#cwg1696-A2-a {{initializing field 'v' with default member initializer}}
      A2(int v) : v(v) {}
      // since-cxx14-warning@-1 {{binding reference member 'v' to stack allocated parameter 'v'}}
      // since-cxx14-note@#cwg1696-A2-a {{reference member declared here}}
      const int &v = 42;  // #cwg1696-A2-a
    };
    A2 a1;    // #cwg1696-A2-b
    
    A2 a2(1); // OK, unfortunately
#endif
  }

  struct A { A(); ~A(); };
#if __cplusplus >= 201103L
  struct B {
    A &&a; // #cwg1696-a
    B() : a{} {}
    // since-cxx11-error@-1 {{reference member 'a' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
    //   since-cxx11-note@#cwg1696-a {{reference member declared here}}
  } b;
#endif

  struct C {
    C();
    const A &a; // #cwg1696-C-a
  };
  C::C() : a(A()) {}
  // expected-error@-1 {{reference member 'a' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
  //   expected-note@#cwg1696-C-a {{reference member declared here}}

#if __cplusplus >= 201103L
  // This is OK in C++14 onwards, per CWG1815, though we don't support that yet:
  //   D1 d1 = {};
  // is equivalent to
  //   D1 d1 = {A()};
  // ... which lifetime-extends the A temporary.
  struct D1 {
  // cxx11-error@-1 {{reference member 'a' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
  //   cxx11-note@#cwg1696-d1 {{in implicit default constructor for 'cwg1696::D1' first required here}}
  //   cxx11-note@#cwg1696-D1-a {{initializing field 'a' with default member initializer}}
    const A &a = A(); // #cwg1696-D1-a
  };
  D1 d1 = {}; // #cwg1696-d1

  struct D2 {
    const A &a = A(); // #cwg1696-D2-a
    D2() {}
    // since-cxx11-error@-1 {{reference member 'a' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
    //   since-cxx11-note@#cwg1696-D2-a {{initializing field 'a' with default member initializer}}
  };

  struct D3 {
  // since-cxx11-error@-1 {{reference member 'a' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
  //   since-cxx11-note@#cwg1696-d3 {{in implicit default constructor for 'cwg1696::D3' first required here}}
  //   since-cxx11-note@#cwg1696-D3-a {{initializing field 'a' with default member initializer}}
    const A &a = A(); // #cwg1696-D3-a
  };
  D3 d3; // #cwg1696-d3

  struct haslist1 {
    std::initializer_list<int> il; // #cwg1696-il-1
    haslist1(int i) : il{i, 2, 3} {}
    // since-cxx11-error@-1 {{backing array for 'std::initializer_list' member 'il' is a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
    //   since-cxx11-note@#cwg1696-il-1 {{'std::initializer_list' member declared here}}
  };

  struct haslist2 {
    std::initializer_list<int> il; // #cwg1696-il-2
    haslist2();
  };
  haslist2::haslist2() : il{1, 2} {}
  // since-cxx11-error@-1 {{backing array for 'std::initializer_list' member 'il' is a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
  //   since-cxx11-note@#cwg1696-il-2 {{'std::initializer_list' member declared here}}

  struct haslist3 {
    std::initializer_list<int> il = {1, 2, 3};
  };

  struct haslist4 {
  // since-cxx11-error@-1 {{backing array for 'std::initializer_list' member 'il' is a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
  //   since-cxx11-note@#cwg1696-hl4 {{in implicit default constructor for 'cwg1696::haslist4' first required here}}
  //   since-cxx11-note@#cwg1696-il-4 {{initializing field 'il' with default member initializer}}
    std::initializer_list<int> il = {1, 2, 3}; // #cwg1696-il-4
  };
  haslist4 hl4; // #cwg1696-hl4

  struct haslist5 {
    std::initializer_list<int> il = {1, 2, 3}; // #cwg1696-il-5
    haslist5() {}
    // since-cxx11-error@-1 {{backing array for 'std::initializer_list' member 'il' is a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
    //   since-cxx11-note@#cwg1696-il-5 {{nitializing field 'il' with default member initializer}}
  };
#endif
}
