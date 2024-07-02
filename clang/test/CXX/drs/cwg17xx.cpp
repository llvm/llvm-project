// RUN: %clang_cc1 -std=c++98 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

namespace cwg1710 { // cwg1710: no
// FIXME: all of the following is well-formed
template <typename T> struct D1 : T::template B<int>::template C<int> {};
template <typename T> struct D2 : T::B<int>::template C<int> {};
// expected-error@-1 {{use 'template' keyword to treat 'B' as a dependent template name}}
template <typename T> struct D3 : T::template B<int>::C<int> {};
// expected-error@-1 {{use 'template' keyword to treat 'C' as a dependent template name}}
template <typename T> struct D4 : T::B<int>::C<int> {};
// expected-error@-1 {{use 'template' keyword to treat 'B' as a dependent template name}}
// expected-error@-2 {{use 'template' keyword to treat 'C' as a dependent template name}}
} // namespace cwg1710

namespace cwg1715 { // cwg1715: 3.9
#if __cplusplus >= 201103L
  struct B {
    template<class T> B(T, typename T::Q);
  };

  class S {
    using Q = int;
    template<class T> friend B::B(T, typename T::Q);
  };

  struct D : B {
    using B::B;
  };
  struct E : B { // #cwg1715-E
    template<class T> E(T t, typename T::Q q) : B(t, q) {} // #cwg1715-E-ctor
  };

  B b(S(), 1);
  D d(S(), 2);
  E e(S(), 3);
  // since-cxx11-error@-1 {{no matching constructor for initialization of 'E'}}
  //   since-cxx11-note@#cwg1715-E-ctor {{candidate template ignored: substitution failure [with T = S]: 'Q' is a private member of 'cwg1715::S'}}
  //   since-cxx11-note@#cwg1715-E {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 2 were provided}}
  //   since-cxx11-note@#cwg1715-E {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 2 were provided}}
#endif
}

namespace cwg1719 { // cwg1719: 19
#if __cplusplus >= 201103L
struct CStruct {
  int one;
  int two;
};

struct CStruct2 {
  int one;
  int two;
};

struct CStructWithQualifiers {
  const int one;
  volatile int two;
};

static_assert(__is_layout_compatible(CStruct, const CStruct2), "");
static_assert(__is_layout_compatible(CStruct, volatile CStruct2), "");
static_assert(__is_layout_compatible(const CStruct, volatile CStruct2), "");
static_assert(__is_layout_compatible(int, const int), "");
static_assert(__is_layout_compatible(int, volatile int), "");
static_assert(__is_layout_compatible(const int, volatile int), "");
static_assert(__is_layout_compatible(CStruct, CStructWithQualifiers), "");
static_assert(__is_layout_compatible(int[], const volatile int[]), "");
#endif
} // namespace cwg1719

namespace cwg1722 { // cwg1722: 9
#if __cplusplus >= 201103L
void f() {
  const auto lambda = [](int x) { return x + 1; };
  // Without the DR applied, this static_assert would fail.
  static_assert(
      noexcept((int (*)(int))(lambda)),
      "Lambda-to-function-pointer conversion is expected to be noexcept");
}
#endif
} // namespace cwg1722

namespace cwg1734 { // cwg1734: no
#if __cplusplus >= 201103L
struct A {
    A(const A&) = delete;
};
// FIXME: 'A' should not be trivially copyable because the class lacks at least
// one non-deleted copy constructor, move constructor, copy assignment
// operator, or move assignment operator.
static_assert(__is_trivially_copyable(A), "");
#endif
}

namespace cwg1736 { // cwg1736: 3.9
#if __cplusplus >= 201103L
struct S {
  template <class T> S(T t) {
    struct L : S {
      using S::S;
    };
    typename T::type value;
    // since-cxx11-error@-1 {{type 'int' cannot be used prior to '::' because it has no members}}
    //   since-cxx11-note@#cwg1736-l {{in instantiation of function template specialization 'cwg1736::S::S<int>' requested here}}
    //   since-cxx11-note@#cwg1736-s {{in instantiation of function template specialization 'cwg1736::S::S<cwg1736::Q>' requested here}}
    L l(value); // #cwg1736-l
  }
};
struct Q { typedef int type; } q;
S s(q); // #cwg1736-s
#endif
}

namespace cwg1738 { // cwg1738: sup P0136R1
#if __cplusplus >= 201103L
struct A {
  template <typename T>
  A(int, T) {}
};

struct B : A {
  using A::A;
};

// FIXME: this is well-formed since P0136R1
template B::B(int, double);
// since-cxx11-error@-1 {{explicit instantiation of 'B' does not refer to a function template, variable template, member function, member class, or static data member}}
#endif
}

// cwg1748 is in cwg1748.cpp

namespace cwg1753 { // cwg1753: 11
  typedef int T;
  struct A { typedef int T; };
  namespace B { typedef int T; }

  void f(T n) {
    n.~T();
    n.T::~T();

    n.cwg1753::~T();
    // expected-error@-1 {{'cwg1753' does not refer to a type name in pseudo-destructor expression; expected the name of type 'T' (aka 'int')}}
    n.cwg1753::T::~T();

    n.A::~T();
    // expected-error@-1 {{the type of object expression ('T' (aka 'int')) does not match the type being destroyed ('A') in pseudo-destructor expression}}
    n.A::T::~T();

    n.B::~T();
    // expected-error@-1 {{'B' does not refer to a type name in pseudo-destructor expression; expected the name of type 'T' (aka 'int')}}
    n.B::T::~T();

  #if __cplusplus >= 201103L
    n.decltype(n)::~T();
    // since-cxx11-error@-1 {{'decltype(n)' (aka 'int') is not a class, namespace, or enumeration}}
    n.T::~decltype(n)();
    // since-cxx11-error@-1 {{expected a class name after '~' to name a destructor}}
    n.~decltype(n)(); // OK
  #endif
  }
}

namespace cwg1756 { // cwg1756: 3.7
#if __cplusplus >= 201103L
  // Direct-list-initialization of a non-class object
  
  int a{0};
  
  struct X { operator int(); } x;
  int b{x};
#endif
}

namespace cwg1758 { // cwg1758: 3.7
#if __cplusplus >= 201103L
  // Explicit conversion in copy/move list initialization

  struct X { X(); };
  struct Y { explicit operator X(); } y;
  X x{y};

  struct A {
    A() {}
    A(const A &) {}
  };
  struct B {
    operator A() { return A(); }
  } b;
  A a{b};
#endif
}

namespace cwg1762 { // cwg1762: 14
#if __cplusplus >= 201103L
  float operator ""_E(const char *);
  float operator ""E(const char *);
  // since-cxx11-error@-1 {{invalid suffix on literal; C++11 requires a space between literal and identifier}}
  // since-cxx11-warning@-2 {{user-defined literal suffixes not starting with '_' are reserved; no literal will invoke this operator}}
#endif
}

// cwg1772 is in cwg177x.cpp

namespace cwg1778 { // cwg1778: 9
  // Superseded by P1286R2.
#if __cplusplus >= 201103L
  struct A { A() noexcept(true) = default; };
  struct B { B() noexcept(false) = default; };
  static_assert(noexcept(A()), "");
  static_assert(!noexcept(B()), "");

  struct C { A a; C() noexcept(false) = default; };
  struct D { B b; D() noexcept(true) = default; };
  static_assert(!noexcept(C()), "");
  static_assert(noexcept(D()), "");
#endif
}

// cwg1779 is in cwg177x.cpp

namespace cwg1794 { // cwg1794: yes
                   // NB: dup 1710
#if __cplusplus >= 201103L
template <template <typename> class Template> struct Internal {
  template <typename Arg> using Bind = Template<Arg>;
};

template <template <typename> class Template, typename Arg>
using Instantiate = Template<Arg>;

template <template <typename> class Template, typename Argument>
using Bind = Instantiate<Internal<Template>::template Bind, Argument>;
#endif
} // namespace cwg1794
