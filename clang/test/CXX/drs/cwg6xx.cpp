// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98-17,cxx98-14,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx11-20,cxx98-17,cxx11-17,cxx98-14,since-cxx11,cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,cxx11-20,cxx98-17,cxx11-17,cxx98-14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,cxx11-20,cxx98-17,cxx11-17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,cxx11-20,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

namespace cwg600 { // cwg600: 2.8
struct S {
  void f(int);

private:
  void f(double); // #cwg600-f-double
};

void g(S *sp) {
  sp->f(2);
  // access control is applied after overload resolution
  sp->f(2.2);
  // expected-error@-1 {{'f' is a private member of 'cwg600::S'}}
  //   expected-note@#cwg600-f-double {{declared private here}}
}
} // namespace cwg600

namespace std {
  struct type_info {};
  __extension__ typedef __SIZE_TYPE__ size_t;
} // namespace std

namespace cwg601 { // cwg601: 2.7
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
static_assert(0x80000000 < -1, "0x80000000 should be unsigned");
#endif

#if MAX > 0xFFFFFFFFFFFFFFFF && 0x8000000000000000 < -1
#error 0x8000000000000000 should be signed
#endif

#if __cplusplus >= 201103L && __LLONG_MAX__ == 0x7FFFFFFFFFFFFFFF
static_assert(0x8000000000000000 < -1, "0x8000000000000000 should be unsigned");
#endif

#undef MAX
} // namespace cwg601

namespace cwg602 { // cwg602: 2.7
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
} // namespace cwg602

namespace cwg603 { // cwg603: 3.1
  template<unsigned char> struct S {};
  typedef S<'\001'> S1;
  typedef S<(1ul << __CHAR_BIT__) + 1> S1;
  // since-cxx11-error@-1 {{non-type template argument evaluates to 257, which cannot be narrowed to type 'unsigned char'}}
} // namespace cwg603

// cwg604: na
// cwg605 is in cwg605.cpp

namespace cwg606 { // cwg606: 3.0
#if __cplusplus >= 201103L
  template<typename T> struct S {};
  template<typename T> void f(S<T> &&); // #cwg606-f
  template<typename T> void g(T &&);
  template<typename T> void h(const T &&); // #cwg606-h

  void test(S<int> s) {
    f(s);
    // since-cxx11-error@-1 {{no matching function for call to 'f'}}
    //   since-cxx11-note@#cwg606-f {{candidate function [with T = int] not viable: expects an rvalue for 1st argument}}
    g(s);
    h(s);
    // since-cxx11-error@-1 {{no matching function for call to 'h'}}
    //   since-cxx11-note@#cwg606-h {{candidate function [with T = cwg606::S<int>] not viable: expects an rvalue for 1st argument}}

    g(test);
    h(test); // ok, an rvalue reference can bind to a function lvalue
  }
#endif
} // namespace cwg606

namespace cwg607 { // cwg607: 2.7
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
} // namespace cwg607

namespace cwg608 { // cwg608: 2.7
  struct A { virtual void f(); };
  struct B : A {};
  struct C : A { void f(); };
  struct D : B, C {};
} // namespace cwg608

namespace cwg610 { // cwg610: 2.7
static_assert(-0u == 0u, "");
} // namespace cwg610

namespace cwg611 { // cwg611: 2.7
  int k;
  struct S { int &r; } s = { k ? k : k };
} // namespace cwg611

// cwg612: na

namespace cwg613 { // cwg613: 3.1 c++11
  // see also n2253
  struct A { int n; static void f(); };
  int f(int);
  struct B { virtual void f(); };
  B &g(int);

  int an1 = sizeof(A::n);
  // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
  int an2 = sizeof(A::n + 1); // valid per cwg850
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
} // namespace cwg613

namespace cwg614 { // cwg614: 2.7
static_assert((-1) / 2 == 0, "");
static_assert((-1) % 2 == -1, "");
} // namespace cwg614

namespace cwg615 { // cwg615: 2.7
  int f();
  static int n = f();
} // namespace cwg615

namespace cwg616 { // cwg616: 4
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
} // namespace cwg616

namespace cwg618 { // cwg618: 2.7
#if (unsigned)-1 > 0
#error wrong
#endif
} // namespace cwg618

namespace cwg619 { // cwg619: 3.4
  extern int x[10];
  struct S { static int x[10]; };

  int x[];
  static_assert(sizeof(x) == sizeof(int) * 10, "");
  extern int x[];
  static_assert(sizeof(x) == sizeof(int) * 10, "");

  int S::x[];
  static_assert(sizeof(S::x) == sizeof(int) * 10, "");

  void f() {
    extern int x[];
    sizeof(x);
    // expected-error@-1 {{invalid application of 'sizeof' to an incomplete type 'int[]'}}
  }
} // namespace cwg619

// cwg620: dup 568

namespace cwg621 { // cwg621: 2.7
  template<typename T> T f();
  template<> int f() {} // #cwg621-f
  template<> int f<int>() {}
  // expected-error@-1 {{redefinition of 'f<int>'}}
  //   expected-note@#cwg621-f {{previous definition is here}}
} // namespace cwg621

// cwg623: na
// FIXME: Add documentation saying we allow invalid pointer values.

// cwg624 needs a libc++abi test.

namespace cwg625 { // cwg625: 2.9
  template<typename T> struct A {};
  A<auto> x = A<int>();
  // cxx98-error@-1 {{'auto' type specifier is a C++11 extension}}
  // expected-error@-2 {{'auto' not allowed in template argument}}
  void f(int);
  void (*p)(auto) = f;
  // cxx98-error@-1 {{'auto' type specifier is a C++11 extension}}
  // expected-error@-2 {{'auto' not allowed in function prototype}}
} // namespace cwg625

namespace cwg626 { // cwg626: 2.7
#define STR(x) #x
  char c[2] = STR(c); // ok, type matches
  wchar_t w[2] = STR(w);
  // expected-error@-1 {{initializing wide char array with non-wide string literal}}
} // namespace cwg626

namespace cwg627 { // cwg627: 2.7
  void f() {
    // FIXME: emitted diagnostic have a room for improvement
    true a = 0;
    // expected-error@-1 {{expected ';' after expression}}
    // expected-error@-2 {{use of undeclared identifier 'a'}}
    // expected-warning@-3 {{expression result unused}}
  }
} // namespace cwg627

// cwg628: na

namespace cwg629 { // cwg629: 2.9
  typedef int T;
  int n = 1;
  void f() {
    auto T = 2; // #cwg629-T
    // cxx98-error@-1 {{expected unqualified-id}}

    auto T(n);
    // since-cxx11-error@-1 {{redefinition of 'T'}}
    //   since-cxx11-note@#cwg629-T {{previous definition is here}}
  }
} // namespace cwg629

namespace cwg630 { // cwg630: 2.7
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
static_assert(!MB_EQ_WC, "__STDC_MB_MIGHT_NEQ_WC__ but all basic source characters have same representation");
#endif
#else
static_assert(MB_EQ_WC, "!__STDC_MB_MIGHT_NEQ_WC__ but some character differs");
#endif
} // namespace cwg630

// cwg631: na

namespace cwg632 { // cwg632: 2.7
  struct S { int n; } s = {{5}};
  // expected-warning@-1 {{braces around scalar initializer}}
} // namespace cwg632

// cwg633: na
// see also n2993

namespace cwg634 { // cwg634: 2.7
  struct S { S(); S(const S&); virtual void f(); ~S(); };
  int f(...);
  char f(int);
  template<typename T> int (&g(T))[sizeof f(T())];
  int (&a)[sizeof(int)] = g(S());
  int (&b)[1] = g(0);
  int k = f(S());
  // cxx98-error@-1 {{cannot pass object of non-POD type 'S' through variadic function; call will abort at runtime}}
  // since-cxx11-error@-2 {{cannot pass object of non-trivial type 'S' through variadic function; call will abort at runtime}}
} // namespace cwg634

namespace cwg635 { // cwg635: 2.7
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
  template<typename T> D<T>::D() {} // #cwg635-D
  template<typename T> template<typename U> D<T>::D() {}
  template<typename T> D<T>::D<T>() {} // #cwg635-D-T
  // expected-error@#cwg635-D-T {{out-of-line constructor for 'D' cannot have template arguments}}
  // expected-error@#cwg635-D-T {{redefinition of 'cwg635::D<T>'}}
  //   expected-note@#cwg635-D {{previous definition is here}}
} // namespace cwg635

namespace cwg637 { // cwg637: 3.0
  void f(int i) {
    i = ++i + 1;
    i = i++ + 1;
    // cxx98-14-warning@-1 {{multiple unsequenced modifications to 'i'}}
  }
} // namespace cwg637

namespace cwg638 { // cwg638: no
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
    // expected-warning@-1 {{dependent nested name specifier 'A<T>' for friend class declaration is not supported; turning off access control for 'X'}}
    template<class T> friend void A<T>::f();
    // expected-warning@-1 {{dependent nested name specifier 'A<T>' for friend class declaration is not supported; turning off access control for 'X'}}
    template<class T> friend void A<T>::g();
    // expected-warning@-1 {{dependent nested name specifier 'A<T>' for friend class declaration is not supported; turning off access control for 'X'}}
    template<class T> friend void A<T>::C::h();
    // expected-warning@-1 {{dependent nested name specifier 'A<T>::C' for friend class declaration is not supported; turning off access control for 'X'}}
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
} // namespace cwg638

namespace cwg639 { // cwg639: 3.3
  void f(int i) {
    void((i = 0) + (i = 0));
    // expected-warning@-1 {{multiple unsequenced modifications to 'i'}}
  }
} // namespace cwg639

namespace cwg641 { // cwg641: 2.7
  namespace std_example {
    struct abc;

    struct xyz {
      xyz(); // #cwg641-xyz-ctor
      xyz(xyz &); // #cwg641-xyz-copy-ctor

      operator xyz &() = delete;
      // expected-warning@-1 {{conversion function converting 'cwg641::std_example::xyz' to itself will never be used}}
      // cxx98-error@-2 {{deleted function definitions are a C++11 extension}}
      operator abc &() = delete;
      // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    };

    struct abc : xyz {};

    template<typename T>
    void use(T &); // #cwg641-use
    void test() {
      use<xyz>(xyz());
      // expected-error@-1 {{no matching function for call to 'use'}}
      //   expected-note@#cwg641-use {{candidate function template not viable: expects an lvalue for 1st argument}}
      use<const xyz>(xyz());
      // cxx98-error@-1 {{no viable constructor copying parameter of type 'xyz'; C++98 requires a copy constructor when binding a reference to a temporary}}
      //   cxx98-note@#cwg641-xyz-copy-ctor {{candidate constructor not viable: expects an lvalue for 1st argument}}
      //   cxx98-note@#cwg641-xyz-ctor {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
    }
  }

  template<typename T> struct error { typedef typename T::error type; };

  struct A {
    template<typename T, typename error<T>::type = 0> operator T() const;
    // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  };
  A a;
  void f(A&); // #cwg641-f
  void g(const A ca) {
    f(A());
    // expected-error@-1 {{no matching function for call to 'f'}}
    //   expected-note@#cwg641-f {{candidate function not viable: expects an lvalue for 1st argument}}
    f(ca);
    // expected-error@-1 {{no matching function for call to 'f'}}
    //   expected-note@#cwg641-f {{candidate function not viable: 1st argument ('const A') would lose const qualifier}}
    (void)A();
    (void)ca;
  }
} // namespace cwg641

namespace cwg642 { // cwg642: 2.7
  void f() {
    const int i = 2;
    {
      char i[i];
      static_assert(sizeof(i) == 2, "");
    }
  }

  struct s { int a; };
  void g(int s) {
    struct s *p = new struct s;
    p->a = s;
  }
} // namespace cwg642

namespace cwg643 { // cwg643: 3.2
#if __cplusplus >= 201103L
  struct A {
    int x;
    auto f() -> decltype(this->x);
    auto f(A &a) -> decltype(a.x);
    auto g() -> decltype(x);
    auto h() -> decltype(this->y);
    // since-cxx11-error@-1 {{no member named 'y' in 'cwg643::A'}}
    auto h(A &a) -> decltype(a.y);
    // since-cxx11-error@-1 {{no member named 'y' in 'cwg643::A'}}
    auto i() -> decltype(y);
    // since-cxx11-error@-1 {{use of undeclared identifier 'y'}}
    int y;
  };
#endif
} // namespace cwg643

namespace cwg644 { // cwg644: partial
#if __cplusplus >= 201103L
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

  // FIXME: According to CWG644, E<C> is a literal type despite having virtual
  // base classes. This appears to be a wording defect.
  template<typename T>
  struct E : T {
    constexpr E() = default;
  };
  static_assert(!__is_literal_type(E<C>), "");
#endif
} // namespace cwg644

// cwg645 increases permission to optimize; it's not clear that it's possible to
// test for this.
// cwg645: na

namespace cwg646 { // cwg646: sup 981
#if __cplusplus >= 201103L
  struct A {
    constexpr A(const A&) = default; // ok
  };

  struct B {
    constexpr B() {}
    B(B&);
  };
  constexpr B b = {}; // ok
#endif
} // namespace cwg646

namespace cwg647 { // cwg647: 3.1
#if __cplusplus >= 201103L
  // This is partially superseded by cwg1358.
  struct A {
    constexpr virtual void f() const;
    constexpr virtual void g() const {}
    // cxx11-17-error@-1 {{virtual function cannot be constexpr}}
  };

  struct X { virtual void f() const; }; // #cwg647-f
  struct B : X {
    constexpr void f() const {}
    // cxx11-17-error@-1 {{virtual function cannot be constexpr}}
    //   cxx11-17-note@#cwg647-f {{overridden virtual function is here}}
  };

  struct NonLiteral { NonLiteral() {} }; // #cwg647-NonLiteral

  struct C {
    constexpr C(NonLiteral);
    constexpr C(NonLiteral, int) {}
    // cxx11-20-error@-1 {{constexpr constructor's 1st parameter type 'NonLiteral' is not a literal type}}
    //   cxx11-20-note@#cwg647-NonLiteral {{'NonLiteral' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
    constexpr C() try {} catch (...) {}
    // cxx11-17-error@-1 {{function try block in constexpr constructor is a C++20 extension}}
    // cxx11-error@-2 {{use of this statement in a constexpr constructor is a C++14 extension}}
  };

  struct D {
    operator int() const;
    constexpr D(int) {}
    D(float); // #cwg647-D-float-ctor
  };
  constexpr int get();
  struct E {
    int n;
    D d;

    // FIXME: We should diagnose this, as the conversion function is not
    // constexpr. However, that part of this issue is supreseded by cwg1364 and
    // others; no diagnostic is required for this any more.
    constexpr E()
        : n(D(0)),
          d(0) {}

    constexpr E(int)
    // cxx11-20-error@-1 {{constexpr constructor never produces a constant expression}}
    //   cxx11-20-note@#cwg647-int-d {{non-constexpr constructor 'D' cannot be used in a constant expression}}
    //   cxx11-20-note@#cwg647-D-float-ctor {{declared here}}
        : n(0),
          d(0.0f) {} // #cwg647-int-d
    constexpr E(float f)
    // cxx11-20-error@-1 {{never produces a constant expression}}
    //   cxx11-20-note@#cwg647-float-d {{non-constexpr constructor}}
    //   cxx11-20-note@#cwg647-D-float-ctor {{declared here}}
        : n(get()),
          d(D(0) + f) {} // #cwg647-float-d
  };
#endif
} // namespace cwg647

namespace cwg648 { // cwg648: 2.7
#if __cplusplus >= 201103L
  int f();
  constexpr int a = (true ? 1 : f());
  constexpr int b = false && f();
  constexpr int c = true || f();
#endif
} // namespace cwg648

namespace cwg649 { // cwg649: 3.5
#if __cplusplus >= 201103L
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
  // This part is superseded by cwg2130 and eventually by aligned allocation support.
  auto *p = new Z;
#endif
} // namespace cwg649

// cwg650 is in cwg650.cpp

namespace cwg651 { // cwg651: 2.7
#if __cplusplus >= 201103L
  struct X {
    virtual X &f();
  };
  struct Y : X {
    Y &f();
  };
  using T = decltype(((X&&)Y()).f());
  using T = X &;
#endif
} // namespace cwg651

namespace cwg652 { // cwg652: 3.1
#if __cplusplus >= 201103L
  constexpr int n = 1.2 * 3.4;
  static_assert(n == 4, "");
#endif
} // namespace cwg652

// cwg653 is in cwg653.cpp

namespace cwg654 { // cwg654: sup 1423
#if __cplusplus >= 201103L
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
#endif
} // namespace cwg654

namespace cwg655 { // cwg655: 3.0
  struct A { A(int); }; // #cwg655-A
  struct B : A {
    A a; // #cwg655-a
    B();
    B(int) : B() {}
    // cxx98-error@-1 {{delegating constructors are permitted only in C++11}}
    B(int*) : A() {} // #cwg655-delegating-to-A
    // expected-error@-1 {{no matching constructor for initialization of 'A'}}
    //   expected-note@#cwg655-A {{candidate constructor not viable: requires 1 argument, but 0 were provided}}
    //   expected-note@#cwg655-A {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 0 were provided}}
    //   since-cxx11-note@#cwg655-A {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 0 were provided}}
    // expected-error@#cwg655-delegating-to-A {{constructor for 'cwg655::B' must explicitly initialize the member 'a' which does not have a default constructor}}
    //   expected-note@#cwg655-a {{member is declared here}}
    //   expected-note@#cwg655-A {{'cwg655::A' declared here}}
  };
} // namespace cwg655

namespace cwg656 { // cwg656: 2.8
  struct A { A(const A&) = delete; };
  // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
  struct B : A {};
  struct X { operator B(); } x;
  const A &r = x;
  struct Y : private A { // #cwg656-Y
    operator B() volatile;
  };
  extern Y y;
  extern volatile Y vy;
  // Conversion not considered due to reference-related types.
  const A &s = y;
  // expected-error@-1 {{cannot cast 'const Y' to its private base class 'const A'}}
  //   expected-note@#cwg656-Y {{declared private here}}
  const A &t = vy;
  // expected-error@-1 {{binding reference of type 'const A' to value of type 'volatile Y' drops 'volatile' qualifier}}

  struct C { operator struct D(); } c;
  struct D : C {};
  const D &d = c; // ok, D not reference-related to C

  template<typename T> void accept(T); // #cwg656-accept-T
  template<typename T> void accept(...) = delete; // #cwg656-accept-var
  // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
  void f() {
    accept<const A&>(x);
    accept<const A&>(y);
    // expected-error@-1 {{cannot cast 'const Y' to its private base class 'const cwg656::A'}}
    //   expected-note@#cwg656-Y {{declared private here}}
    accept<const A&>(vy); // #cwg656-vy
    // expected-error@-1 {{call to deleted function 'accept'}}
    //   expected-note@#cwg656-accept-var {{candidate function [with T = const cwg656::A &] has been explicitly deleted}}
    //   expected-note@#cwg656-accept-T {{candidate function template not viable: no known conversion from 'volatile Y' to 'const A &' for 1st argument}}
    // expected-error@#cwg656-vy {{no matching constructor for initialization of 'volatile Y'}}
    //   expected-note@#cwg656-Y {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('volatile Y') would lose volatile qualifier}}
    //   expected-note@#cwg656-Y {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
    accept<const D&>(c);
  }
} // namespace cwg656

namespace cwg657 { // cwg657: partial
  struct Abs { virtual void x() = 0; }; // #cwg657-Abs
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
  //   expected-note@#cwg657-Abs {{unimplemented pure virtual method 'x' in 'Abs'}}
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
} // namespace cwg657

// cwg658 is in cwg658.cpp

namespace cwg659 { // cwg659: 3.0
#if __cplusplus >= 201103L
  static_assert(alignof(char) == alignof(char&), "");
  static_assert(alignof(int) == alignof(int&), "");
  int n = alignof(int(&)());
  // since-cxx11-error@-1 {{invalid application of 'alignof' to a function type}}
  struct A; // #cwg659-A
  int m = alignof(A&);
  // since-cxx11-error@-1 {{invalid application of 'alignof' to an incomplete type 'A'}}
  //   since-cxx11-note@#cwg659-A {{forward declaration of 'cwg659::A'}}
#endif
} // namespace cwg659

namespace cwg660 { // cwg660: 3.0
#if __cplusplus >= 201103L
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
#endif
} // namespace cwg660

// cwg661 is in cwg661.cpp

namespace cwg662 { // cwg662: 2.7
  template <typename T> void f(T t) {
    T &tr = t;
    T *tp = &t;
    // expected-error@-1 {{'tp' declared as a pointer to a reference of type 'int &'}}
    //   expected-note@#cwg662-f-call {{in instantiation of function template specialization 'cwg662::f<int &>' requested here}}
#if __cplusplus >= 201103L
    auto *ap = &t;
#endif
  }
  void g(int n) { f<int&>(n); } // #cwg662-f-call
} // namespace cwg662

namespace cwg663 { // cwg663: sup P1949
  int ЍЎ = 123;
} // namespace cwg663

namespace cwg664 { // cwg664: 2.7
#if __cplusplus >= 201103L
  struct A { A(const A&) = delete; };
  A &&f(A &&a, int n) {
    if (n)
      return f(static_cast<A&&>(a), n - 1);
    return static_cast<A&&>(a);
  }
#endif
} // namespace cwg664

namespace cwg665 { // cwg665: 2.8
  struct A { virtual ~A(); };
  struct B : A {} *b;
  struct C : private A {} *c; // #cwg665-C
  struct D : B, C {} *d;

  struct VB : virtual A {} *vb;
  struct VC : private virtual A {} *vc; // #cwg665-VC
  struct VD : VB, VC {} *vd;

  void f() {
    (void)dynamic_cast<A*>(b);
    (void)dynamic_cast<A*>(c);
    // expected-error@-1 {{cannot cast 'cwg665::C' to its private base class 'cwg665::A'}}
    //   expected-note@#cwg665-C {{declared private here}}
    (void)dynamic_cast<A*>(d);
    /* expected-error@-1 {{ambiguous conversion from derived class 'cwg665::D' to base class 'cwg665::A':
    struct cwg665::D -> B -> A
    struct cwg665::D -> C -> A}} */
    (void)dynamic_cast<A*>(vb);
    (void)dynamic_cast<A*>(vc); // emitting diagnostic, even though it could be valid at runtime
    // expected-error@-1 {{cannot cast 'cwg665::VC' to its private base class 'cwg665::A'}}
    //   expected-note@#cwg665-VC {{declared private here}}
    (void)dynamic_cast<A*>(vd);
  }
} // namespace cwg665

namespace cwg666 { // cwg666: 2.8
  struct P { friend P operator*(P, P); P(int); } p(0);

  template<int> int f();
  template<typename T> int f() {
    T::type *p = 0;
    // expected-error@-1 {{missing 'typename' prior to dependent type name 'cwg666::Y::type'}}
    //   expected-note@#cwg666-f-Y {{in instantiation of function template specialization 'cwg666::f<cwg666::Y>' requested here}}
    int a(T::type);
    // expected-error@-1 {{missing 'typename' prior to dependent type name 'cwg666::Y::type'}}
    return f<T::type>();
    // expected-error@-1 {{missing 'typename' prior to dependent type name 'cwg666::Y::type'}}
  }
  struct X { static const int type = 0; };
  struct Y { typedef int type; };
  int a = f<X>();
  int b = f<Y>(); // #cwg666-f-Y
} // namespace cwg666

// Triviality is entirely different in C++98.
namespace cwg667 { // cwg667: 8
#if __cplusplus >= 201103L
  struct A {
    A() = default; // #cwg667-A-ctor
    // since-cxx11-warning@-1 {{explicitly defaulted default constructor is implicitly deleted}}
    //   since-cxx11-note@#cwg667-r {{default constructor of 'A' is implicitly deleted because field 'r' of reference type 'int &' would not be initialized}}
    //   since-cxx11-note@#cwg667-A-ctor {{replace 'default' with 'delete'}}
    int &r; // #cwg667-r
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
#endif
} // namespace cwg667

// cwg668 needs an libc++abi test

namespace cwg669 { // cwg669: 3.1
#if __cplusplus >= 201103L
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
#endif
} // namespace cwg669

namespace cwg671 { // cwg671: 2.9
  enum class E { e };
  // cxx98-error@-1 {{scoped enumerations are a C++11 extension}}
  E e = static_cast<E>(0);
  int n = static_cast<int>(E::e);
  // cxx98-error@-1 {{use of enumeration in a nested name specifier is a C++11 extension}}
  int m = static_cast<int>(e);
} // namespace cwg671

// cwg672 is in cwg672.cpp

namespace cwg673 { // cwg673: 2.7
  template<typename> struct X { static const int n = 0; };

  class A {
    friend class B *f();
    class C *f();
    void f(class D *);
    enum { e = X<struct E>::n };
    void g() { extern struct FF *p; }
  };
  B *b;
  C *c;
  D *d;
  E *e;
  FF *ff;
  // expected-error@-1 {{unknown type name 'FF'}}
} // namespace cwg673

namespace cwg674 { // cwg674: 8
  template<typename T> int f(T);

  int g(int);
  template<typename T> int g(T);

  int h(int);
  template<typename T> int h(T);

  class X {
    friend int cwg674::f(int);
    friend int cwg674::g(int);
    friend int cwg674::h<>(int);
    int n; // #cwg674-X-n
  };

  template<typename T> int f(T) { return X().n; }
  int g(int) { return X().n; }
  template<typename T> int g(T) { return X().n; }
  // expected-error@-1 {{'n' is a private member of 'cwg674::X'}}
  //   expected-note@#cwg674-g-int {{in instantiation of function template specialization 'cwg674::g<int>' requested here}}
  //   expected-note@#cwg674-X-n {{implicitly declared private here}}
  int h(int) { return X().n; }
  // expected-error@-1 {{'n' is a private member of 'cwg674::X'}}
  //   expected-note@#cwg674-X-n {{implicitly declared private here}}
  template<typename T> int h(T) { return X().n; }

  template int f(int);
  template int g(int); // #cwg674-g-int
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
    int n; // #cwg674-Z-n
  };

  template<typename T> int Y::f(T) { return Z().n; }
  int Y::g(int) { return Z().n; }
  template<typename T> int Y::g(T) { return Z().n; }
  // expected-error@-1 {{'n' is a private member of 'cwg674::Z'}}
  //   expected-note@#cwg674-Y-g-int {{in instantiation of function template specialization 'cwg674::Y::g<int>' requested here}}
  //   expected-note@#cwg674-Z-n {{implicitly declared private here}}
  int Y::h(int) { return Z().n; }
  // expected-error@-1 {{'n' is a private member of 'cwg674::Z'}}
  //   expected-note@#cwg674-Z-n {{implicitly declared private here}}
  template<typename T> int Y::h(T) { return Z().n; }

  // FIXME: Should the <> be required here?
  template int Y::f<>(int);
  template int Y::g<>(int); // #cwg674-Y-g-int
  template int Y::h<>(int);
} // namespace cwg674

namespace cwg675 { // cwg675: dup 739
  template<typename T> struct A { T n : 1; };
#if __cplusplus >= 201103L
  static_assert(A<char>{1}.n < 0, "");
  // since-cxx11-warning@-1 {{implicit truncation from 'int' to a one-bit wide bit-field changes value from 1 to -1}}
  static_assert(A<int>{1}.n < 0, "");
  // since-cxx11-warning@-1 {{implicit truncation from 'int' to a one-bit wide bit-field changes value from 1 to -1}}
  static_assert(A<long long>{1}.n < 0, "");
  // since-cxx11-warning@-1 {{implicit truncation from 'int' to a one-bit wide bit-field changes value from 1 to -1}}
#endif
} // namespace cwg675

// cwg676: na

namespace cwg677 { // cwg677: no
  struct A {
    void *operator new(std::size_t);
    void operator delete(void*) = delete; // #cwg677-A-delete
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
  };
  struct B {
    void *operator new(std::size_t);
    void operator delete(void*) = delete; // #cwg677-B-delete
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    virtual ~B();
  };
  void f(A *p) { delete p; }
  // expected-error@-1 {{attempt to use a deleted function}}
  //   expected-note@#cwg677-A-delete {{'operator delete' has been explicitly marked deleted here}}
  // FIXME: This appears to be valid; we shouldn't even be looking up the 'operator delete' here.
  void f(B *p) { delete p; }
  // expected-error@-1 {{attempt to use a deleted function}}
  //   expected-note@#cwg677-B-delete {{'operator delete' has been explicitly marked deleted here}}
  B::~B() {}
  // expected-error@-1 {{attempt to use a deleted function}}
  //   expected-note@#cwg677-B-delete {{'operator delete' has been explicitly marked deleted here}}
} // namespace cwg677

// cwg678 FIXME: check that the modules ODR check catches this

namespace cwg679 { // cwg679: 2.7
  struct X {};
  template<int> void operator+(X, X);
  template<> void operator+<0>(X, X) {} // #cwg679-def
  template<> void operator+<0>(X, X) {}
  // expected-error@-1 {{redefinition of 'operator+<0>'}}
  //   expected-note@#cwg679-def {{previous definition is here}}
} // namespace cwg679

// cwg680: na

namespace cwg681 { // cwg681: partial
#if __cplusplus >= 201103L
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
#endif
} // namespace cwg681

namespace cwg683 { // cwg683: 3.3
#if __cplusplus >= 201103L
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
#endif
} // namespace cwg683

namespace cwg684 { // cwg684: sup 1454
#if __cplusplus >= 201103L
  void f() {
    int a;  // #cwg684-a
    constexpr int *p = &a;
    // since-cxx11-error@-1 {{constexpr variable 'p' must be initialized by a constant expression}}
    //   since-cxx11-note@-2 {{pointer to 'a' is not a constant expression}}
    //   since-cxx11-note@#cwg684-a {{here}}
  }
#endif
} // namespace cwg684

namespace cwg685 { // cwg685: 10
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

  int j(unsigned int); // #cwg685-j-uint
  void j(long); // #cwg685-j-long
  int d = j(g);
  // expected-error@-1 {{call to 'j' is ambiguous}}
  //   expected-note@#cwg685-j-uint {{candidate function}}
  //   expected-note@#cwg685-j-long {{candidate function}}

  // Valid per cwg1601
  int k(short);
  void k(int);
  int x = k(g);
} // namespace cwg685

namespace cwg686 { // cwg686: 3.0
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
    (void)new struct G*; // #cwg686-G
    (void)new struct G{}*; // #cwg686-G-def
    // expected-error@-1 {{allocation of incomplete type 'struct G'}}
    //   expected-note@#cwg686-G {{forward declaration of 'G'}}
    // since-cxx11-error@#cwg686-G-def {{expected expression}}
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
#if __cplusplus <= 201402L
    void g() throw(struct Q);
    // cxx98-14-error@-1 {{incomplete type 'struct Q' is not allowed in exception specification}}
    //   cxx98-14-note@-2 {{forward declaration of 'Q'}}
    void h() throw(struct Q {});
    // cxx98-14-error@-1 {{'Q' cannot be defined in a type specifier}}
#endif
  }
  template<struct R *> struct X;
  template<struct R {} *> struct Y;
  // expected-error@-1 {{'cwg686::R' cannot be defined in a type specifier}}
} // namespace cwg686

namespace cwg687 { // cwg687 (9 c++20, but the issue is still considered open)
  template<typename T> void f(T a) {
    // This is valid in C++20.
    g<int>(a);
    // cxx98-17-error@-1 {{use of function template name with no prior declaration in function call with explicit template arguments is a C++20 extension}}

    // This is not.
    template g<int>(a);
    // expected-error@-1 {{expected '<' after 'template'}}
  }
} // namespace cwg687

namespace cwg692 { // cwg692: 16
  // Also see cwg1395.

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
    template <typename T, typename U> void f(U, A1<U, T> *p = 0) = delete; // #cwg692-f-deleted
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    template <typename U> int &f(U, A1<U, U> *p = 0); // #cwg692-f
    template <typename T> void g(T, T = T()); // #cwg692-g
    template <typename T, typename... U> void g(T, U...); // #cwg692-g-variadic
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
      //   expected-note@#cwg692-f-deleted {{candidate function [with T = int, U = int] has been explicitly deleted}}
      //   expected-note@#cwg692-f {{candidate function [with U = int]}}
      g(42);
      // expected-error@-1 {{ambiguous}}
      //   expected-note@#cwg692-g {{candidate function [with T = int]}}
      //   expected-note@#cwg692-g-variadic {{candidate function [with T = int, U = <>]}}
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
      // This is made ambiguous by cwg692, but made valid again by cwg1395.
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
} // namespace cwg692

namespace cwg696 { // cwg696: 3.1
  void f(const int*);
  void g() {
    const int N = 10; // #cwg696-N
    struct A {
      void h() {
        int arr[N]; (void)arr;
        f(&N);
        // expected-error@-1 {{reference to local variable 'N' declared in enclosing function 'cwg696::g'}}
        //   expected-note@#cwg696-N {{'N' declared here}}
      }
    };
#if __cplusplus >= 201103L
    (void) [] { int arr[N]; (void)arr; };
    (void)[] { f(&N); };
    // since-cxx11-error@-1 {{variable 'N' cannot be implicitly captured in a lambda with no capture-default specified}}
    //   since-cxx11-note@#cwg696-N {{'N' declared here}}
    //   since-cxx11-note@-3 {{lambda expression begins here}}
    //   since-cxx11-note@-4 {{capture 'N' by value}}
    //   since-cxx11-note@-5 {{capture 'N' by reference}}
    //   since-cxx11-note@-6 {{default capture by value}}
    //   since-cxx11-note@-7 {{default capture by reference}}
#endif
  }
} // namespace cwg696
