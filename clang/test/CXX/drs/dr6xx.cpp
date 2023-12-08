// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98-17,cxx98-14,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx98-17,cxx11-17,cxx98-14,since-cxx11,cxx11 -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,cxx98-17,cxx11-17,cxx98-14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,cxx98-17,cxx11-17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking

namespace dr600 { // dr600: 2.8
struct S {
  void f(int);

private:
  void f(double); // #dr600-f-double
};

void g(S *sp) {
  sp->f(2);
  // access control is applied after overload resolution
  sp->f(2.2);
  // expected-error@-1 {{'f' is a private member of 'dr600::S'}}
  //   expected-note@#dr600-f-double {{declared private here}}
}
} // namespace dr600

namespace std {
  struct type_info {};
  __extension__ typedef __SIZE_TYPE__ size_t;
} // namespace std

namespace dr601 { // dr601: yes
#if __cplusplus >= 201103L
#define MAX __LLONG_MAX__
#else
#define MAX __LONG_MAX__
#endif

#if 0x8000 < -1
#error 0x8000 should be signed
#endif

#if MAX > 0xFFFFFFFF && 0x80000000 < -1
#error 0x80000000 should be signed
#endif

#if __INT_MAX__ == 0x7FFFFFFF
_Static_assert(0x80000000 < -1, "0x80000000 should be unsigned");
// expected-error@-1 {{'_Static_assert' is a C11 extension}}
#endif

#if MAX > 0xFFFFFFFFFFFFFFFF && 0x8000000000000000 < -1
#error 0x8000000000000000 should be signed
#endif

#if __cplusplus >= 201103L && __LLONG_MAX__ == 0x7FFFFFFFFFFFFFFF
static_assert(0x8000000000000000 < -1, "0x8000000000000000 should be unsigned");
#endif

#undef MAX
}

namespace dr602 { // dr602: yes
  template<class T> struct A {
    template<class U> friend struct A;
  };

  template<class T> struct B {
    class C {
      template<class U> friend struct B;
      typedef int type;
    };
    typename C::type ct; // ok, befriended
  };
  B<int> b;
}

namespace dr603 { // dr603: yes
  template<unsigned char> struct S {};
  typedef S<'\001'> S1;
  typedef S<(1ul << __CHAR_BIT__) + 1> S1;
  // since-cxx11-error@-1 {{cannot be narrowed}}
}

// dr604: na
// dr605 needs IRGen test

namespace dr606 { // dr606: 3.0
#if __cplusplus >= 201103L
  template<typename T> struct S {};
  template<typename T> void f(S<T> &&); // #dr606-f
  template<typename T> void g(T &&);
  template<typename T> void h(const T &&); // #dr606-h

  void test(S<int> s) {
    f(s);
    // since-cxx11-error@-1 {{no matching function for call to 'f'}}
    //   since-cxx11-note@#dr606-f {{candidate function [with T = int] not viable: expects an rvalue for 1st argument}}
    g(s);
    h(s);
    // since-cxx11-error@-1 {{no matching function for call to 'h'}}
    //   since-cxx11-note@#dr606-h {{candidate function [with T = dr606::S<int>] not viable: expects an rvalue for 1st argument}}

    g(test);
    h(test); // ok, an rvalue reference can bind to a function lvalue
  }
#endif
}

namespace dr607 { // dr607: yes
namespace example1 {
struct Y {};

template <typename T> struct X : public virtual Y {};

template <typename T> class A : public X<T> {
  template <typename S> A(S) : S() {}
};

template A<int>::A(Y);
} // namespace example1

namespace example2 {
namespace N {
struct B {
  B(int);
};
typedef B typedef_B;
struct D : B {
  D();
};
} // namespace N

N::D::D() : typedef_B(0) {}
} // namespace example2
} // namespace dr607

namespace dr608 { // dr608: yes
  struct A { virtual void f(); };
  struct B : A {};
  struct C : A { void f(); };
  struct D : B, C {};
}

int dr610[-0u == 0u ? 1 : -1]; // dr610: yes

namespace dr611 { // dr611: yes
  int k;
  struct S { int &r; } s = { k ? k : k };
}

// dr612: na

namespace dr613 { // dr613: yes c++11
  // see also n2253
  struct A { int n; static void f(); };
  int f(int);
  struct B { virtual void f(); };
  B &g(int);

  int an1 = sizeof(A::n);
  // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
  int an2 = sizeof(A::n + 1); // valid per dr850
  // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
  int an3 = sizeof A::n;
  // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
  int an4 = sizeof(f(A::n));
  // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
  int an5 = sizeof(g(A::n));
  // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
  const std::type_info &an6 = typeid(A::n);
  // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
  const std::type_info &an7 = typeid(A::n + 1);
  // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
  const std::type_info &an8 = typeid(f(A::n));
  // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
  const std::type_info &an9 = typeid(g(A::n));
  // expected-error@-1 {{invalid use of non-static data member 'n'}}

  void A::f() {
    int an1 = sizeof n;
    // cxx98-error@-1 {{invalid use of member 'n' in static member function}}
    const std::type_info &an2 = typeid(n + 1);
    // cxx98-error@-1 {{invalid use of member 'n' in static member function}}
    const std::type_info &an3 = typeid(g(n));
    // cxx98-error@-1 {{invalid use of member 'n' in static member function}}
    // since-cxx11-error@-2 {{invalid use of non-static data member 'n'}}
  }
}

int dr614_a[(-1) / 2 == 0 ? 1 : -1]; // dr614: yes
int dr614_b[(-1) % 2 == -1 ? 1 : -1];

namespace dr615 { // dr615: yes
  int f();
  static int n = f();
}

namespace dr616 { // dr616: 4
#if __cplusplus >= 201103L
  struct S { int n; } s;
  S f();
  using T = decltype((S().n));
  using T = decltype((static_cast<S&&>(s).n));
  using T = decltype((f().n));
  using T = decltype(S().*&S::n);
  using T = decltype(static_cast<S&&>(s).*&S::n);
  using T = decltype(f().*&S::n);
  using T = int&&;

  using U = decltype(S().n);
  using U = decltype(static_cast<S&&>(s).n);
  using U = int;
#endif
}

namespace dr618 { // dr618: yes
#if (unsigned)-1 > 0
#error wrong
#endif
}

namespace dr619 { // dr619: yes
  extern int x[10];
  struct S { static int x[10]; };

  int x[];
  _Static_assert(sizeof(x) == sizeof(int) * 10, "");
  // expected-error@-1 {{'_Static_assert' is a C11 extension}}
  extern int x[];
  _Static_assert(sizeof(x) == sizeof(int) * 10, "");
  // expected-error@-1 {{'_Static_assert' is a C11 extension}}

  int S::x[];
  _Static_assert(sizeof(S::x) == sizeof(int) * 10, "");
  // expected-error@-1 {{'_Static_assert' is a C11 extension}}

  void f() {
    extern int x[];
    sizeof(x);
    // expected-error@-1 {{invalid application of 'sizeof' to an incomplete type 'int[]'}}
  }
}

// dr620: dup 568

namespace dr621 { // dr621: yes
  template<typename T> T f();
  template<> int f() {} // #dr621-f
  template<> int f<int>() {}
  // expected-error@-1 {{redefinition of 'f<int>'}}
  //   expected-note@#dr621-f {{previous definition is here}}
}

// dr623: na
// FIXME: Add documentation saying we allow invalid pointer values.

// dr624 needs an IRGen check.

namespace dr625 { // dr625: yes
  template<typename T> struct A {};
  A<auto> x = A<int>();
  // cxx98-error@-1 {{'auto' type specifier is a C++11 extension}}
  // expected-error@-2 {{'auto' not allowed in template argument}}
  void f(int);
  void (*p)(auto) = f;
  // cxx98-error@-1 {{'auto' type specifier is a C++11 extension}}
  // expected-error@-2 {{'auto' not allowed in function prototype}}
}

namespace dr626 { // dr626: yes
#define STR(x) #x
  char c[2] = STR(c); // ok, type matches
  wchar_t w[2] = STR(w);
  // expected-error@-1 {{initializing wide char array with non-wide string literal}}
}

namespace dr627 { // dr627: yes
  void f() {
    // FIXME: emitted diagnostic have a room for improvement
    true a = 0;
    // expected-error@-1 {{expected ';' after expression}}
    // expected-error@-2 {{use of undeclared identifier 'a'}}
    // expected-warning@-3 {{expression result unused}}
  }
}

// dr628: na

namespace dr629 { // dr629: 2.9
  typedef int T;
  int n = 1;
  void f() {
    auto T = 2; // #dr629-T
    // cxx98-error@-1 {{expected unqualified-id}}

    auto T(n);
    // since-cxx11-error@-1 {{redefinition of 'T'}}
    //   since-cxx11-note@#dr629-T {{previous definition is here}}
  }
}

namespace dr630 { // dr630: yes
const bool MB_EQ_WC =
    ' ' == L' ' && '\t' == L'\t' && '\v' == L'\v' && '\r' == L'\r' &&
    '\n' == L'\n' && //
    'a' == L'a' && 'b' == L'b' && 'c' == L'c' && 'd' == L'd' && 'e' == L'e' &&
    'f' == L'f' && 'g' == L'g' && 'h' == L'h' && 'i' == L'i' && 'j' == L'j' &&
    'k' == L'k' && 'l' == L'l' && 'm' == L'm' && 'n' == L'n' && 'o' == L'o' &&
    'p' == L'p' && 'q' == L'q' && 'r' == L'r' && 's' == L's' && 't' == L't' &&
    'u' == L'u' && 'v' == L'v' && 'w' == L'w' && 'x' == L'x' && 'y' == L'y' &&
    'z' == L'z' && //
    'A' == L'A' && 'B' == L'B' && 'C' == L'C' && 'D' == L'D' && 'E' == L'E' &&
    'F' == L'F' && 'G' == L'G' && 'H' == L'H' && 'I' == L'I' && 'J' == L'J' &&
    'K' == L'K' && 'L' == L'L' && 'M' == L'M' && 'N' == L'N' && 'O' == L'O' &&
    'P' == L'P' && 'Q' == L'Q' && 'R' == L'R' && 'S' == L'S' && 'T' == L'T' &&
    'U' == L'U' && 'V' == L'V' && 'W' == L'W' && 'X' == L'X' && 'Y' == L'Y' &&
    'Z' == L'Z' && //
    '0' == L'0' && '1' == L'1' && '2' == L'2' && '3' == L'3' && '4' == L'4' &&
    '5' == L'5' && '6' == L'6' && '7' == L'7' && '8' == L'8' &&
    '9' == L'9' && //
    '_' == L'_' && '{' == L'{' && '}' == L'}' && '[' == L'[' && ']' == L']' &&
    '#' == L'#' && '(' == L'(' && ')' == L')' && '<' == L'<' && '>' == L'>' &&
    '%' == L'%' && ':' == L':' && ';' == L';' && '.' == L'.' && '?' == L'?' &&
    '*' == L'*' && '+' == L'+' && '-' == L'-' && '/' == L'/' && '^' == L'^' &&
    '&' == L'&' && '|' == L'|' && '~' == L'~' && '!' == L'!' && '=' == L'=' &&
    ',' == L',' && '\\' == L'\\' && '"' == L'"' && '\'' == L'\'';
#if __STDC_MB_MIGHT_NEQ_WC__
#ifndef __FreeBSD__ // PR22208, FreeBSD expects us to give a bad (but conforming) answer here.
_Static_assert(!MB_EQ_WC, "__STDC_MB_MIGHT_NEQ_WC__ but all basic source characters have same representation");
// expected-error@-1 {{'_Static_assert' is a C11 extension}}
#endif
#else
_Static_assert(MB_EQ_WC, "!__STDC_MB_MIGHT_NEQ_WC__ but some character differs");
// expected-error@-1 {{'_Static_assert' is a C11 extension}}
#endif
}

// dr631: na

namespace dr632 { // dr632: yes
  struct S { int n; } s = {{5}};
  // expected-warning@-1 {{braces around scalar initializer}}
}

// dr633: na
// see also n2993

namespace dr634 { // dr634: yes
  struct S { S(); S(const S&); virtual void f(); ~S(); };
  int f(...);
  char f(int);
  template<typename T> int (&g(T))[sizeof f(T())];
  int (&a)[sizeof(int)] = g(S());
  int (&b)[1] = g(0);
  int k = f(S());
  // cxx98-error@-1 {{cannot pass object of non-POD type 'S' through variadic function; call will abort at runtime}}
  // since-cxx11-error@-2 {{cannot pass object of non-trivial type 'S' through variadic function; call will abort at runtime}}
}

namespace dr635 { // dr635: yes
  template<typename T> struct A { A(); ~A(); };
  template<typename T> A<T>::A<T>() {}
  // expected-error@-1 {{out-of-line constructor for 'A' cannot have template arguments}}
  template<typename T> A<T>::~A<T>() {}

  template<typename T> struct B { B(); ~B(); };
  template<typename T> B<T>::B() {}
  template<typename T> B<T>::~B() {}

  struct C { template<typename T> C(); C(); };
  template<typename T> C::C() {}
  C::C() {}
  template<> C::C<int>() {}
  // expected-error@-1 {{qualified reference to 'C' is a constructor name rather than a type in this context}}
  // expected-error@-2 {{expected unqualified-id}}
  /*FIXME: needed for error recovery:*/;

  template<typename T> struct D { template<typename U> D(); D(); };
  template<typename T> D<T>::D() {} // #dr635-D
  template<typename T> template<typename U> D<T>::D() {}
  template<typename T> D<T>::D<T>() {} // #dr635-D-T
  // expected-error@#dr635-D-T {{out-of-line constructor for 'D' cannot have template arguments}}
  // expected-error@#dr635-D-T {{redefinition of 'D<T>'}}
  //   expected-note@#dr635-D {{previous definition is here}}
}

namespace dr637 { // dr637: yes
  void f(int i) {
    i = ++i + 1;
    i = i++ + 1;
    // cxx98-14-warning@-1 {{multiple unsequenced modifications to 'i'}}
  }
}

namespace dr638 { // dr638: no
  template<typename T> struct A {
    struct B;
    void f();
    void g();
    struct C {
      void h();
    };
  };

  class X {
    typedef int type;
    template<class T> friend struct A<T>::B; 
    // expected-warning@-1 {{dependent nested name specifier 'A<T>::' for friend class declaration is not supported; turning off access control for 'X'}}
    template<class T> friend void A<T>::f();
    // expected-warning@-1 {{dependent nested name specifier 'A<T>::' for friend class declaration is not supported; turning off access control for 'X'}}
    template<class T> friend void A<T>::g();
    // expected-warning@-1 {{dependent nested name specifier 'A<T>::' for friend class declaration is not supported; turning off access control for 'X'}}
    template<class T> friend void A<T>::C::h();
    // expected-warning@-1 {{dependent nested name specifier 'A<T>::C::' for friend class declaration is not supported; turning off access control for 'X'}}
  };

  template<> struct A<int> {
    X::type a; // FIXME: private
    struct B {
      X::type b; // ok
    };
    int f() { X::type c; } // FIXME: private
    void g() { X::type d; } // ok
    struct D {
      void h() { X::type e; } // FIXME: private
    };
  };
}

namespace dr639 { // dr639: 3.3
  void f(int i) {
    void((i = 0) + (i = 0));
    // expected-warning@-1 {{multiple unsequenced modifications to 'i'}}
  }
}

namespace dr641 { // dr641: yes
  namespace std_example {
    struct abc;

    struct xyz {
      xyz(); // #dr641-xyz-ctor
      xyz(xyz &); // #dr641-xyz-copy-ctor

      operator xyz &() = delete;
      // expected-warning@-1 {{conversion function converting 'dr641::std_example::xyz' to itself will never be used}}
      // cxx98-error@-2 {{deleted function definitions are a C++11 extension}}
      operator abc &() = delete;
      // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    };

    struct abc : xyz {};

    template<typename T>
    void use(T &); // #dr641-use
    void test() {
      use<xyz>(xyz());
      // expected-error@-1 {{no matching function for call to 'use'}}
      //   expected-note@#dr641-use {{candidate function template not viable: expects an lvalue for 1st argument}}
      use<const xyz>(xyz());
      // cxx98-error@-1 {{no viable constructor copying parameter of type 'xyz'; C++98 requires a copy constructor when binding a reference to a temporary}}
      //   cxx98-note@#dr641-xyz-copy-ctor {{candidate constructor not viable: expects an lvalue for 1st argument}}
      //   cxx98-note@#dr641-xyz-ctor {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
    }
  }

  template<typename T> struct error { typedef typename T::error type; };

  struct A {
    template<typename T, typename error<T>::type = 0> operator T() const;
    // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  };
  A a;
  void f(A&); // #dr641-f
  void g(const A ca) {
    f(A());
    // expected-error@-1 {{no matching function for call to 'f'}}
    //   expected-note@#dr641-f {{candidate function not viable: expects an lvalue for 1st argument}}
    f(ca);
    // expected-error@-1 {{no matching function for call to 'f'}}
    //   expected-note@#dr641-f {{candidate function not viable: 1st argument ('const A') would lose const qualifier}}
    (void)A();
    (void)ca;
  }
}

namespace dr642 { // dr642: yes
  void f() {
    const int i = 2;
    {
      char i[i];
      _Static_assert(sizeof(i) == 2, "");
      // expected-error@-1 {{'_Static_assert' is a C11 extension}}
    }
  }

  struct s { int a; };
  void g(int s) {
    struct s *p = new struct s;
    p->a = s;
  }
}

#if __cplusplus >= 201103L
namespace dr643 { // dr643: 3.2
  struct A {
    int x;
    auto f() -> decltype(this->x);
    auto f(A &a) -> decltype(a.x);
    auto g() -> decltype(x);
    auto h() -> decltype(this->y);
    // since-cxx11-error@-1 {{no member named 'y' in 'dr643::A'}}
    auto h(A &a) -> decltype(a.y);
    // since-cxx11-error@-1 {{no member named 'y' in 'dr643::A'}}
    auto i() -> decltype(y);
    // since-cxx11-error@-1 {{use of undeclared identifier 'y'}}
    int y;
  };
}
#endif

#if __cplusplus >= 201103L
namespace dr644 { // dr644: partial
  struct A {
    A() = default;
    int x, y;
  };
  static_assert(__is_literal_type(A), "");

  struct B : A {};
  static_assert(__is_literal_type(B), "");

  struct C : virtual A {};
  static_assert(!__is_literal_type(C), "");

  struct D { C c; };
  static_assert(!__is_literal_type(D), "");

  // FIXME: According to DR644, E<C> is a literal type despite having virtual
  // base classes. This appears to be a wording defect.
  template<typename T>
  struct E : T {
    constexpr E() = default;
  };
  static_assert(!__is_literal_type(E<C>), "");
}
#endif

// dr645 increases permission to optimize; it's not clear that it's possible to
// test for this.
// dr645: na

#if __cplusplus >= 201103L
namespace dr646 { // dr646: sup 981
  struct A {
    constexpr A(const A&) = default; // ok
  };

  struct B {
    constexpr B() {}
    B(B&);
  };
  constexpr B b = {}; // ok
}
#endif

#if __cplusplus >= 201103L
namespace dr647 { // dr647: 3.1
  // This is partially superseded by dr1358.
  struct A {
    constexpr virtual void f() const;
    constexpr virtual void g() const {}
    // cxx11-17-error@-1 {{virtual function cannot be constexpr}}
  };

  struct X { virtual void f() const; }; // #dr647-f
  struct B : X {
    constexpr void f() const {}
    // cxx11-17-error@-1 {{virtual function cannot be constexpr}}
    //   cxx11-17-note@#dr647-f {{overridden virtual function is here}}
  };

  struct NonLiteral { NonLiteral() {} }; // #dr647-NonLiteral

  struct C {
    constexpr C(NonLiteral);
    constexpr C(NonLiteral, int) {}
    // since-cxx11-error@-1 {{constexpr constructor's 1st parameter type 'NonLiteral' is not a literal type}}
    //   since-cxx11-note@#dr647-NonLiteral {{'NonLiteral' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
    constexpr C() try {} catch (...) {}
    // cxx11-17-error@-1 {{function try block in constexpr constructor is a C++20 extension}}
    // cxx11-error@-2 {{use of this statement in a constexpr constructor is a C++14 extension}}
  };

  struct D {
    operator int() const;
    constexpr D(int) {}
    D(float); // #dr647-D-float-ctor
  };
  constexpr int get();
  struct E {
    int n;
    D d;

    // FIXME: We should diagnose this, as the conversion function is not
    // constexpr. However, that part of this issue is supreseded by dr1364 and
    // others; no diagnostic is required for this any more.
    constexpr E()
        : n(D(0)),
          d(0) {}

    constexpr E(int)
    // since-cxx11-error@-1 {{constexpr constructor never produces a constant expression}}
    //   since-cxx11-note@#dr647-int-d {{non-constexpr constructor 'D' cannot be used in a constant expression}}
    //   since-cxx11-note@#dr647-D-float-ctor {{declared here}}
        : n(0),
          d(0.0f) {} // #dr647-int-d
    constexpr E(float f)
    // since-cxx11-error@-1 {{never produces a constant expression}}
    //   since-cxx11-note@#dr647-float-d {{non-constexpr constructor}}
    //   since-cxx11-note@#dr647-D-float-ctor {{declared here}}
        : n(get()),
          d(D(0) + f) {} // #dr647-float-d
  };
}
#endif

#if __cplusplus >= 201103L
namespace dr648 { // dr648: yes
  int f();
  constexpr int a = (true ? 1 : f());
  constexpr int b = false && f();
  constexpr int c = true || f();
}
#endif

#if __cplusplus >= 201103L
namespace dr649 { // dr649: 3.5
// Maximum alignment is 8192 bytes for Windows, and 4 GB for Linux
alignas(0x200000000) int n;
// since-cxx11-error-re@-1 {{{{requested alignment must be (8192|4294967296) bytes or smaller}}}}
struct alignas(0x200000000) X {};
// since-cxx11-error-re@-1 {{{{requested alignment must be (8192|4294967296) bytes or smaller}}}}
struct Y {
  int n alignas(0x200000000);
  // since-cxx11-error-re@-1 {{{{requested alignment must be (8192|4294967296) bytes or smaller}}}}
};
  struct alignas(256) Z {};
  // This part is superseded by dr2130 and eventually by aligned allocation support.
  auto *p = new Z;
}
#endif

// dr650 FIXME: add codegen test

#if __cplusplus >= 201103L
namespace dr651 { // dr651: yes
  struct X {
    virtual X &f();
  };
  struct Y : X {
    Y &f();
  };
  using T = decltype(((X&&)Y()).f());
  using T = X &;
}
#endif

#if __cplusplus >= 201103L
namespace dr652 { // dr652: yes
  constexpr int n = 1.2 * 3.4;
  static_assert(n == 4, "");
}
#endif

// dr653 FIXME: add codegen test

#if __cplusplus >= 201103L
namespace dr654 { // dr654: sup 1423
  void f() {
    if (nullptr) {}
    // since-cxx11-warning@-1 {{implicit conversion of nullptr constant to 'bool'}}
    bool b = nullptr;
    // since-cxx11-error@-1 {{cannot initialize a variable of type 'bool' with an rvalue of type 'std::nullptr_t'}}
    if (nullptr == 0) {}
    if (nullptr != 0) {}
    if (nullptr <= 0) {}
    // since-cxx11-error@-1 {{invalid operands to binary expression ('std::nullptr_t' and 'int')}}
    if (nullptr == 1) {}
    // since-cxx11-error@-1 {{invalid operands to binary expression ('std::nullptr_t' and 'int')}}
    if (!nullptr) {}
    // since-cxx11-warning@-1 {{implicit conversion of nullptr constant to 'bool'}}
    decltype(nullptr) n = 0;
    static_cast<int>(nullptr);
    // since-cxx11-error@-1 {{static_cast from 'std::nullptr_t' to 'int' is not allowed}}
    (void)static_cast<decltype(nullptr)>(0);
    static_cast<decltype(nullptr)>(1);
    // since-cxx11-error@-1 {{static_cast from 'int' to 'decltype(nullptr)' (aka 'std::nullptr_t') is not allowed}}
    void(true ? nullptr : 0);
    void(true ? 0 : nullptr);
  }
}
#endif

namespace dr655 { // dr655: yes
  struct A { A(int); }; // #dr655-A
  struct B : A {
    A a; // #dr655-a
    B();
    B(int) : B() {}
    // cxx98-error@-1 {{delegating constructors are permitted only in C++11}}
    B(int*) : A() {} // #dr655-delegating-to-A
    // expected-error@-1 {{no matching constructor for initialization of 'A'}}
    //   expected-note@#dr655-A {{candidate constructor not viable: requires 1 argument, but 0 were provided}}
    //   expected-note@#dr655-A {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 0 were provided}}
    //   since-cxx11-note@#dr655-A {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 0 were provided}}
    // expected-error@#dr655-delegating-to-A {{constructor for 'dr655::B' must explicitly initialize the member 'a' which does not have a default constructor}}
    //   expected-note@#dr655-a {{member is declared here}}
    //   expected-note@#dr655-A {{'dr655::A' declared here}}
  };
}

namespace dr656 { // dr656: yes
  struct A { A(const A&) = delete; };
  // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
  struct B : A {};
  struct X { operator B(); } x;
  const A &r = x;
  struct Y : private A { // #dr656-Y
    operator B() volatile;
  };
  extern Y y;
  extern volatile Y vy;
  // Conversion not considered due to reference-related types.
  const A &s = y;
  // expected-error@-1 {{cannot cast 'const Y' to its private base class 'const A'}}
  //   expected-note@#dr656-Y {{declared private here}}
  const A &t = vy;
  // expected-error@-1 {{binding reference of type 'const A' to value of type 'volatile Y' drops 'volatile' qualifier}}

  struct C { operator struct D(); } c;
  struct D : C {};
  const D &d = c; // ok, D not reference-related to C

  template<typename T> void accept(T); // #dr656-accept-T
  template<typename T> void accept(...) = delete; // #dr656-accept-var
  // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
  void f() {
    accept<const A&>(x);
    accept<const A&>(y);
    // expected-error@-1 {{cannot cast 'const Y' to its private base class 'const dr656::A'}}
    //   expected-note@#dr656-Y {{declared private here}}
    accept<const A&>(vy); // #dr656-vy
    // expected-error@-1 {{call to deleted function 'accept'}}
    //   expected-note@#dr656-accept-var {{candidate function [with T = const dr656::A &] has been explicitly deleted}}
    //   expected-note@#dr656-accept-T {{candidate function template not viable: no known conversion from 'volatile Y' to 'const A &' for 1st argument}}
    // expected-error@#dr656-vy {{no matching constructor for initialization of 'volatile Y'}}
    //   expected-note@#dr656-Y {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('volatile Y') would lose volatile qualifier}}
    //   expected-note@#dr656-Y {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
    accept<const D&>(c);
  }
}

namespace dr657 { // dr657: partial
  struct Abs { virtual void x() = 0; }; // #dr657-Abs
  struct Der : public Abs { virtual void x(); };

  struct Cnvt { template<typename F> Cnvt(F); };

  void foo(Cnvt a);
  void foo(Abs &a);
  void f(Abs *a) { foo(*a); }

  void bar(Abs &a);
  template<typename T> void bar(T);
  void g(Abs *a) { bar(*a); }

  // FIXME: The following examples demonstrate that we might be accepting the
  // above cases for the wrong reason.

  struct C { C(Abs) {} };
  // expected-error@-1 {{parameter type 'Abs' is an abstract class}}
  // expected-note@#dr657-Abs {{unimplemented pure virtual method 'x' in 'Abs'}}
  struct Q { operator Abs() { __builtin_unreachable(); } } q;
  // expected-error@-1 {{return type 'Abs' is an abstract class}}
#if __cplusplus >= 201703L
  // FIXME: We should *definitely* reject this.
  C c = Q().operator Abs();
#endif

  template<typename F> struct Cnvt2 { Cnvt2(F); typedef int type; };

  // FIXME: We should reject this.
  void baz(Abs &a);
  template<typename T> typename Cnvt2<T>::type baz(T);
  void h(Abs *a) { baz(*a); }

  // FIXME: We should reject this too.
  Cnvt2<Abs>::type err;
}

// dr658 FIXME: add codegen test

#if __cplusplus >= 201103L
namespace dr659 { // dr659: 3.0
  static_assert(alignof(char) == alignof(char&), "");
  static_assert(alignof(int) == alignof(int&), "");
  int n = alignof(int(&)());
  // since-cxx11-error@-1 {{invalid application of 'alignof' to a function type}}
  struct A; // #dr659-A
  int m = alignof(A&);
  // since-cxx11-error@-1 {{invalid application of 'alignof' to an incomplete type 'A'}}
  // since-cxx11-note@#dr659-A {{forward declaration of 'dr659::A'}}
}
#endif

#if __cplusplus >= 201103L
namespace dr660 { // dr660: 3.0
  enum : int { a };
  enum class { b };
  // since-cxx11-error@-1 {{scoped enumeration requires a name}}
  auto x = a;

  struct X {
    enum : int { a };
    enum class { b };
    // since-cxx11-error@-1 {{scoped enumeration requires a name}}
  };
  auto y = X::a;
}
#endif

// dr661 FIXME: add codegen test

namespace dr662 { // dr662: yes
  template <typename T> void f(T t) {
    T &tr = t;
    T *tp = &t;
    // expected-error@-1 {{'tp' declared as a pointer to a reference of type 'int &'}}
    // expected-note@#dr662-f-call {{in instantiation of function template specialization 'dr662::f<int &>' requested here}}
#if __cplusplus >= 201103L
    auto *ap = &t;
#endif
  }
  void g(int n) { f<int&>(n); } // #dr662-f-call
}

namespace dr663 { // dr663: sup P1949
  int ЍЎ = 123;
}

#if __cplusplus >= 201103L
namespace dr664 { // dr664: yes
  struct A { A(const A&) = delete; };
  A &&f(A &&a, int n) {
    if (n)
      return f(static_cast<A&&>(a), n - 1);
    return static_cast<A&&>(a);
  }
}
#endif

namespace dr665 { // dr665: 2.8
  struct A { virtual ~A(); };
  struct B : A {} *b;
  struct C : private A {} *c; // #dr665-C
  struct D : B, C {} *d;

  struct VB : virtual A {} *vb;
  struct VC : private virtual A {} *vc; // #dr665-VC
  struct VD : VB, VC {} *vd;

  void f() {
    (void)dynamic_cast<A*>(b);
    (void)dynamic_cast<A*>(c);
    // expected-error@-1 {{cannot cast 'dr665::C' to its private base class 'dr665::A'}}
    //   expected-note@#dr665-C {{declared private here}}
    (void)dynamic_cast<A*>(d);
    /* expected-error@-1 {{ambiguous conversion from derived class 'dr665::D' to base class 'dr665::A':
    struct dr665::D -> B -> A
    struct dr665::D -> C -> A}} */
    (void)dynamic_cast<A*>(vb);
    (void)dynamic_cast<A*>(vc); // emitting diagnostic, even though it could be valid at runtime
    // expected-error@-1 {{cannot cast 'dr665::VC' to its private base class 'dr665::A'}}
    //   expected-note@#dr665-VC {{declared private here}}
    (void)dynamic_cast<A*>(vd);
  }
}

namespace dr666 { // dr666: 2.8
  struct P { friend P operator*(P, P); P(int); } p(0);

  template<int> int f();
  template<typename T> int f() {
    T::type *p = 0;
    // expected-error@-1 {{missing 'typename' prior to dependent type name 'Y::type'}}
    //   expected-note@#dr666-f-Y {{in instantiation of function template specialization 'dr666::f<dr666::Y>' requested here}}
    int a(T::type);
    // expected-error@-1 {{missing 'typename' prior to dependent type name 'Y::type'}}
    return f<T::type>();
    // expected-error@-1 {{missing 'typename' prior to dependent type name 'Y::type'}}
  }
  struct X { static const int type = 0; };
  struct Y { typedef int type; };
  int a = f<X>();
  int b = f<Y>(); // #dr666-f-Y
}

// Triviality is entirely different in C++98.
#if __cplusplus >= 201103L
namespace dr667 { // dr667: 8
  struct A {
    A() = default; // #dr667-A-ctor
    // since-cxx11-warning@-1 {{explicitly defaulted default constructor is implicitly deleted}}
    //   since-cxx11-note@#dr667-r {{default constructor of 'A' is implicitly deleted because field 'r' of reference type 'int &' would not be initialized}}
    //   since-cxx11-note@#dr667-A-ctor {{replace 'default' with 'delete'}}
    int &r; // #dr667-r
  };
  static_assert(!__is_trivially_constructible(A), "");

  struct B { ~B() = delete; };
  union C { B b; };
  static_assert(!__is_trivially_destructible(C), "");

  struct D { D(const D&) = delete; };
  struct E : D {};
  static_assert(!__is_trivially_constructible(E, const E&), "");

  struct F { F &operator=(F&&) = delete; };
  struct G : F {};
  static_assert(!__is_trivially_assignable(G, G&&), "");
}
#endif

// dr668 FIXME: add codegen test

#if __cplusplus >= 201103L
namespace dr669 { // dr669: yes
  void f() {
    int n;
    using T = decltype(n);
    using T = int;
    using U = decltype((n));
    using U = int &;

    [=] {
      using V = decltype(n);
      using V = int;
      using W = decltype((n));
      using W = const int&;
    } ();

    struct X {
      int n;
      void f() const {
        using X = decltype(n);
        using X = int;
        using Y = decltype((n));
        using Y = const int&;
      }
    };
  }
}
#endif

namespace dr671 { // dr671: 2.9
  enum class E { e };
  // cxx98-error@-1 {{scoped enumerations are a C++11 extension}}
  E e = static_cast<E>(0);
  int n = static_cast<int>(E::e);
  // cxx98-error@-1 {{use of enumeration in a nested name specifier is a C++11 extension}}
  int m = static_cast<int>(e);
}

// dr672 FIXME: add codegen test

namespace dr673 { // dr673: yes
  template<typename> struct X { static const int n = 0; };

  class A {
    friend class B *f();
    class C *f();
    void f(class D *);
    enum { e = X<struct E>::n };
    void g() { extern struct F *p; }
  };
  B *b;
  C *c;
  D *d;
  E *e;
  F *f;
  // expected-error@-1 {{unknown type name 'F'}}
}

namespace dr674 { // dr674: 8
  template<typename T> int f(T);

  int g(int);
  template<typename T> int g(T);

  int h(int);
  template<typename T> int h(T);

  class X {
    friend int dr674::f(int);
    friend int dr674::g(int);
    friend int dr674::h<>(int);
    int n; // #dr674-X-n
  };

  template<typename T> int f(T) { return X().n; }
  int g(int) { return X().n; }
  template<typename T> int g(T) { return X().n; }
  // expected-error@-1 {{'n' is a private member of 'dr674::X'}}
  //   expected-note@#dr674-g-int {{in instantiation of function template specialization 'dr674::g<int>' requested here}}
  //   expected-note@#dr674-X-n {{implicitly declared private here}}
  int h(int) { return X().n; }
  // expected-error@-1 {{'n' is a private member of 'dr674::X'}}
  //   expected-note@#dr674-X-n {{implicitly declared private here}}
  template<typename T> int h(T) { return X().n; }

  template int f(int);
  template int g(int); // #dr674-g-int
  template int h(int);


  struct Y {
    template<typename T> int f(T);

    int g(int);
    template<typename T> int g(T);

    int h(int);
    template<typename T> int h(T);
  };

  class Z {
    friend int Y::f(int);
    friend int Y::g(int);
    friend int Y::h<>(int);
    int n; // #dr674-Z-n
  };

  template<typename T> int Y::f(T) { return Z().n; }
  int Y::g(int) { return Z().n; }
  template<typename T> int Y::g(T) { return Z().n; }
  // expected-error@-1 {{'n' is a private member of 'dr674::Z'}}
  //   expected-note@#dr674-Y-g-int {{in instantiation of function template specialization 'dr674::Y::g<int>' requested here}}
  //   expected-note@#dr674-Z-n {{implicitly declared private here}}
  int Y::h(int) { return Z().n; }
  // expected-error@-1 {{'n' is a private member of 'dr674::Z'}}
  //   expected-note@#dr674-Z-n {{implicitly declared private here}}
  template<typename T> int Y::h(T) { return Z().n; }

  // FIXME: Should the <> be required here?
  template int Y::f<>(int);
  template int Y::g<>(int); // #dr674-Y-g-int
  template int Y::h<>(int);
}

namespace dr675 { // dr675: dup 739
  template<typename T> struct A { T n : 1; };
#if __cplusplus >= 201103L
  static_assert(A<char>{1}.n < 0, "");
  // since-cxx11-warning@-1 {{implicit truncation from 'int' to a one-bit wide bit-field changes value from 1 to -1}}
  static_assert(A<int>{1}.n < 0, "");
  // since-cxx11-warning@-1 {{implicit truncation from 'int' to a one-bit wide bit-field changes value from 1 to -1}}
  static_assert(A<long long>{1}.n < 0, "");
  // since-cxx11-warning@-1 {{implicit truncation from 'int' to a one-bit wide bit-field changes value from 1 to -1}}
#endif
}

// dr676: na

namespace dr677 { // dr677: no
  struct A {
    void *operator new(std::size_t);
    void operator delete(void*) = delete; // #dr677-A-delete
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}} 
  };
  struct B {
    void *operator new(std::size_t);
    void operator delete(void*) = delete; // #dr677-B-delete
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    virtual ~B();
  };
  void f(A *p) { delete p; }
  // expected-error@-1 {{attempt to use a deleted function}}
  //   expected-note@#dr677-A-delete {{'operator delete' has been explicitly marked deleted here}}
  // FIXME: This appears to be valid; we shouldn't even be looking up the 'operator delete' here.
  void f(B *p) { delete p; }
  // expected-error@-1 {{attempt to use a deleted function}}
  //   expected-note@#dr677-B-delete {{'operator delete' has been explicitly marked deleted here}}
  B::~B() {}
  // expected-error@-1 {{attempt to use a deleted function}}
  //   expected-note@#dr677-B-delete {{'operator delete' has been explicitly marked deleted here}}
}

// dr678 FIXME: check that the modules ODR check catches this

namespace dr679 { // dr679: yes
  struct X {};
  template<int> void operator+(X, X);
  template<> void operator+<0>(X, X) {} // #dr679-def
  template<> void operator+<0>(X, X) {}
  // expected-error@-1 {{redefinition of 'operator+<0>'}}
  //   expected-note@#dr679-def {{previous definition is here}}
}

// dr680: na

#if __cplusplus >= 201103L
namespace dr681 { // dr681: partial
  auto *a() -> int;
  // since-cxx11-error@-1 {{function with trailing return type must specify return type 'auto', not 'auto *'}}
  auto (*b)() -> int;
  // FIXME: The errors here aren't great.
  auto (*c()) -> int;
  // since-cxx11-error@-1 {{expected function body after function declarator}}
  auto ((*d)()) -> int;
  // since-cxx11-error@-1 {{declaration of variable 'd' with deduced type 'auto ((*)())' requires an initializer}}
  // since-cxx11-error@-2 {{expected ';' after top level declarator}}

  // FIXME: This is definitely wrong. This should be
  //   "function of () returning pointer to function of () returning int"
  // not a function with a deduced return type.
  auto (*e())() -> int;
  // cxx11-error@-1 {{'auto' return without trailing return type; deduced return types are a C++14 extension}}

  auto f() -> int (*)();
  auto g() -> auto (*)() -> int;
}
#endif

#if __cplusplus >= 201103L
namespace dr683 { // dr683: yes
  struct A {
    A() = default;
    A(const A&) = default;
    A(A&);
  };
  static_assert(__is_trivially_constructible(A, const A&), "");
  static_assert(!__is_trivially_constructible(A, A&), "");
  static_assert(!__is_trivial(A), "");

  struct B : A {};
  static_assert(__is_trivially_constructible(B, const B&), "");
  static_assert(__is_trivially_constructible(B, B&), "");
  static_assert(__is_trivial(B), "");
}
#endif

#if __cplusplus >= 201103L
namespace dr684 { // dr684: sup 1454
  void f() {
    int a;  // #dr684-a
    constexpr int *p = &a;
    // expected-error@-1 {{constexpr variable 'p' must be initialized by a constant expression}}
    //   expected-note@-2 {{pointer to 'a' is not a constant expression}}
    //   expected-note@#dr684-a {{here}}
  }
}
#endif

namespace dr685 { // dr685: yes
  enum E : long { e };
  // cxx98-error@-1 {{enumeration types with a fixed underlying type are a C++11 extension}}
  void f(int);
  int f(long);
  int a = f(e);

  enum G : short { g };
  // cxx98-error@-1 {{enumeration types with a fixed underlying type are a C++11 extension}}
  int h(short);
  void h(long);
  int b = h(g);

  int i(int);
  void i(long);
  int c = i(g);

  int j(unsigned int); // #dr685-j-uint
  void j(long); // #dr685-j-long
  int d = j(g);
  // expected-error@-1 {{call to 'j' is ambiguous}}
  //   expected-note@#dr685-j-uint {{candidate function}}
  //   expected-note@#dr685-j-long {{candidate function}}

  // Valid per dr1601
  int k(short);
  void k(int);
  int x = k(g);
}

namespace dr686 { // dr686: 3.0
  void f() {
    (void)dynamic_cast<struct A*>(0);
    // expected-error@-1 {{'A' is an incomplete type}}
    //   expected-note@-2 {{forward declaration of 'A'}}
    (void)dynamic_cast<struct A{}*>(0);
    // expected-error@-1 {{'A' cannot be defined in a type specifier}}
    (void)typeid(struct B*);
    (void)typeid(struct B{}*);
    // expected-error@-1 {{'B' cannot be defined in a type specifier}}
    (void)static_cast<struct C*>(0);
    (void)static_cast<struct C{}*>(0);
    // expected-error@-1 {{'C' cannot be defined in a type specifier}}
    (void)reinterpret_cast<struct D*>(0);
    (void)reinterpret_cast<struct D{}*>(0);
    // expected-error@-1 {{'D' cannot be defined in a type specifier}}
    (void)const_cast<struct E*>(0);
    // expected-error@-1 {{const_cast from 'int' to 'struct E *' is not allowed}}
    (void)const_cast<struct E{}*>(0);
    // expected-error@-1 {{'E' cannot be defined in a type specifier}}
    (void)sizeof(struct F*);
    (void)sizeof(struct F{}*);
    // expected-error@-1 {{'F' cannot be defined in a type specifier}}
    (void)new struct G*; // #dr686-G
    (void)new struct G{}*; // #dr686-G-def
    // expected-error@-1 {{allocation of incomplete type 'struct G'}}
    //   expected-note@#dr686-G {{forward declaration of 'G'}}
    // since-cxx11-error@#dr686-G-def {{expected expression}}
#if __cplusplus >= 201103L
    (void)alignof(struct H*);
    (void)alignof(struct H{}*);
    // since-cxx11-error@-1 {{'H' cannot be defined in a type specifier}}
#endif
    (void)(struct I*)0;
    (void)(struct I{}*)0;
    // expected-error@-1 {{'I' cannot be defined in a type specifier}}
    if (struct J *p = 0) {}
    if (struct J {} *p = 0) {}
    // expected-error@-1 {{'J' cannot be defined in a condition}}
    for (struct K *p = 0; struct L *q = 0; ) {}
    for (struct K {} *p = 0; struct L {} *q = 0; ) {}
    // expected-error@-1 {{'L' cannot be defined in a condition}}
#if __cplusplus >= 201103L
    using M = struct {};
#endif
    struct N {
      operator struct O{}(){};
      // expected-error@-1 {{'N::O' cannot be defined in a type specifier}}
    };
    try {}
    catch (struct P *) {}
    // expected-error@-1 {{cannot catch pointer to incomplete type 'struct P'}}
    //   expected-note@-2 {{forward declaration of 'P'}}
    catch (struct P {} *) {}
    // expected-error@-1 {{'P' cannot be defined in a type specifier}}
#if __cplusplus < 201703L
    void g() throw(struct Q);
    // cxx98-17-error@-1 {{incomplete type 'struct Q' is not allowed in exception specification}}
    //   cxx98-17-note@-2 {{forward declaration of 'Q'}}
    void h() throw(struct Q {});
    // cxx98-17-error@-1 {{'Q' cannot be defined in a type specifier}}
#endif
  }
  template<struct R *> struct X;
  template<struct R {} *> struct Y;
  // expected-error@-1 {{'dr686::R' cannot be defined in a type specifier}}
}

namespace dr687 { // dr687 (9 c++20, but the issue is still considered open)
  template<typename T> void f(T a) {
    // This is valid in C++20.
    g<int>(a);
    // cxx98-17-error@-1 {{use of function template name with no prior declaration in function call with explicit template arguments is a C++20 extension}}

    // This is not.
    template g<int>(a);
    // expected-error@-1 {{expected expression}}
  }
}

namespace dr692 { // dr692: 16
  // Also see dr1395.

  namespace temp_func_order_example2 {
    template <typename... T> struct A1 {};
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template <typename U, typename... T> struct A2 {};
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template <typename T1, typename... U> void e1(A1<T1, U...>) = delete;
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    // cxx98-error@-2 {{deleted function definitions are a C++11 extension}}
    template <typename T1> void e1(A1<T1>);
    template <typename T1, typename... U> void e2(A2<T1, U...>) = delete;
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    // cxx98-error@-2 {{deleted function definitions are a C++11 extension}}
    template <typename T1> void e2(A2<T1>);
    template <typename T, typename U> void f(U, A1<U, T> *p = 0) = delete; // #dr692-f-deleted
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    template <typename U> int &f(U, A1<U, U> *p = 0); // #dr692-f
    template <typename T> void g(T, T = T()); // #dr692-g
    template <typename T, typename... U> void g(T, U...); // #dr692-g-variadic
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    void h() {
      A1<int, int> a;
      int &r = f<int>(42, &a);
      A1<int> b1;
      e1(b1);
      A2<int> b2;
      e2(b2);
      f<int>(42);
      // expected-error@-1 {{call to 'f' is ambiguous}}
      // expected-note@#dr692-f-deleted {{candidate function [with T = int, U = int] has been explicitly deleted}}
      // expected-note@#dr692-f {{candidate function [with U = int]}}
      g(42);
      // expected-error@-1 {{ambiguous}}
      // expected-note@#dr692-g {{candidate function [with T = int]}}
      // expected-note@#dr692-g-variadic {{candidate function [with T = int, U = <>]}}
    }
  }

  namespace temp_func_order_example3 {
    template <typename T, typename... U> void f(T, U...);
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template <typename T> void f(T);
    template <typename T, typename... U> int &g(T *, U...);
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template <typename T> void g(T);
    void h(int i) {
      // This is made ambiguous by dr692, but made valid again by dr1395.
      f(&i);
      int &r = g(&i);
    }
  }

  namespace temp_deduct_partial_example {
    template <typename... Args> char &f(Args... args);
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template <typename T1, typename... Args> short &f(T1 a1, Args... args);
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template <typename T1, typename T2> int &f(T1 a1, T2 a2);
    void g() {
      char &a = f();
      short &b = f(1, 2, 3);
      int &c = f(1, 2);
    }
  }

  namespace temp_deduct_type_example1 {
    template <class T1, class ...Z> class S;
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template <class T1, class ...Z> class S<T1, const Z&...>;
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template <class T1, class T2> class S<T1, const T2&> {};
    S<int, const int&> s;

    template<class T, class... U> struct A;
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template<class T1, class T2, class... U> struct A<T1,T2*,U...> {};
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template<class T1, class T2> struct A<T1,T2>;
    template struct A<int, int*>;
  }

  namespace temp_deduct_type_example3 {
    template<class T, class... U> void f(T*, U...){}
    // cxx98-error@-1 {{variadic templates are a C++11 extension}}
    template<class T> void f(T){}
    template void f(int*);
  }
}

namespace dr696 { // dr696: 3.1
  void f(const int*);
  void g() {
    const int N = 10; // #dr696-N
    struct A {
      void h() {
        int arr[N]; (void)arr;
        f(&N);
        // expected-error@-1 {{reference to local variable 'N' declared in enclosing function 'dr696::g'}}
        // expected-note@#dr696-N {{'N' declared here}}
      }
    };
#if __cplusplus >= 201103L
    (void) [] { int arr[N]; (void)arr; };
    (void)[] { f(&N); };
    // since-cxx11-error@-1 {{variable 'N' cannot be implicitly captured in a lambda with no capture-default specified}}
    //   since-cxx11-note@#dr696-N {{'N' declared here}}
    //   since-cxx11-note@-3 {{lambda expression begins here}}
    //   since-cxx11-note@-4 {{capture 'N' by value}}
    //   since-cxx11-note@-5 {{capture 'N' by reference}}
    //   since-cxx11-note@-6 {{default capture by value}}
    //   since-cxx11-note@-7 {{default capture by reference}}
#endif
  }
}
