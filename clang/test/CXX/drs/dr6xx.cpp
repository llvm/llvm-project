// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace std { struct type_info {}; }

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
_Static_assert(0x80000000 < -1, "0x80000000 should be unsigned"); // expected-error {{C11}}
#endif

#if MAX > 0xFFFFFFFFFFFFFFFF && 0x8000000000000000 < -1
#error 0x8000000000000000 should be signed
#endif

#if __cplusplus >= 201103L && __LLONG_MAX__ == 0x7FFFFFFFFFFFFFFF
static_assert(0x8000000000000000 < -1, "0x8000000000000000 should be unsigned"); // expected-error {{C11}}
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
#if __cplusplus >= 201103L
  // expected-error@-2 {{cannot be narrowed}}
#endif
}

// dr604: na
// dr605 needs IRGen test

namespace dr606 { // dr606: yes
#if __cplusplus >= 201103L
  template<typename T> struct S {};
  template<typename T> void f(S<T> &&); // expected-note {{no known conversion from 'S<int>' to 'S<int> &&'}}
  template<typename T> void g(T &&);
  template<typename T> void h(const T &&); // expected-note {{no known conversion from 'S<int>' to 'const dr606::S<int> &&'}}

  void test(S<int> s) {
    f(s); // expected-error {{no match}}
    g(s);
    h(s); // expected-error {{no match}}

    g(test);
    h(test); // ok, an rvalue reference can bind to a function lvalue
  }
#endif
}

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
  int an2 = sizeof(A::n + 1); // valid per dr850
  int an3 = sizeof A::n;
  int an4 = sizeof(f(A::n));
  int an5 = sizeof(g(A::n));
  const std::type_info &an6 = typeid(A::n);
  const std::type_info &an7 = typeid(A::n + 1);
  const std::type_info &an8 = typeid(f(A::n));
  const std::type_info &an9 = typeid(g(A::n)); // expected-error {{non-static}}
#if __cplusplus < 201103L
  // expected-error@-10 {{non-static}}
  // expected-error@-10 {{non-static}}
  // expected-error@-10 {{non-static}}
  // expected-error@-10 {{non-static}}
  // expected-error@-10 {{non-static}}
  // expected-error@-10 {{non-static}}
  // expected-error@-10 {{non-static}}
  // expected-error@-10 {{non-static}}
#endif

  void A::f() {
    int an1 = sizeof n;
    const std::type_info &an2 = typeid(n + 1);
#if __cplusplus < 201103L
  // expected-error@-3 {{static}}
  // expected-error@-3 {{static}}
#endif
    const std::type_info &an3 = typeid(g(n)); // expected-error {{static}}
  }
}

int dr614_a[(-1) / 2 == 0 ? 1 : -1]; // dr614: yes
int dr614_b[(-1) % 2 == -1 ? 1 : -1];

namespace dr615 { // dr615: yes
  int f();
  static int n = f();
}

namespace dr616 { // dr616: no
#if __cplusplus >= 201103L
  struct S { int n; } s;
  // FIXME: These should all be 'int &&'
  using T = decltype(S().n);
  using T = decltype(static_cast<S&&>(s).n);
  using T = decltype(S().*&S::n); // expected-note 2{{previous}}
  using T = decltype(static_cast<S&&>(s).*&S::n); // expected-error {{different type}}
  using T = int&&; // expected-error {{different type}}
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
  _Static_assert(sizeof(x) == sizeof(int) * 10, ""); // expected-error {{C11}}
  extern int x[];
  _Static_assert(sizeof(x) == sizeof(int) * 10, ""); // expected-error {{C11}}

  int S::x[];
  _Static_assert(sizeof(S::x) == sizeof(int) * 10, ""); // expected-error {{C11}}

  void f() {
    extern int x[];
    sizeof(x); // expected-error {{incomplete}}
  }
}

// dr620: dup 568

namespace dr621 {
  template<typename T> T f();
  template<> int f() {} // expected-note {{previous}}
  template<> int f<int>() {} // expected-error {{redefinition}}
}

// dr623: na
// FIXME: Add documentation saying we allow invalid pointer values.

// dr624 needs an IRGen check.

namespace dr625 { // dr625: yes
  template<typename T> struct A {};
  A<auto> x = A<int>(); // expected-error {{'auto' not allowed in template argument}} expected-error 0-1{{extension}}
  void f(int);
  void (*p)(auto) = f; // expected-error {{'auto' not allowed in function prototype}} expected-error 0-1{{extension}}
}

namespace dr626 { // dr626: yes
#define STR(x) #x
  char c[2] = STR(c); // ok, type matches
  wchar_t w[2] = STR(w); // expected-error {{initializing wide char array with non-wide string literal}}
}

namespace dr627 { // dr627: yes
  void f() {
    true a = 0; // expected-error +{{}} expected-warning {{unused}}
  }
}

// dr628: na

namespace dr629 { // dr629: yes
  typedef int T;
  int n = 1;
  void f() {
    auto T = 2;
#if __cplusplus < 201103L
    // expected-error@-2 {{expected unqualified-id}}
#else
    // expected-note@-4 {{previous}}
#endif

    auto T(n);
#if __cplusplus >= 201103L
    // expected-error@-2 {{redefinition of 'T'}}
#endif
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
_Static_assert(!MB_EQ_WC, "__STDC_MB_MIGHT_NEQ_WC__ but all basic source characters have same representation"); // expected-error {{C11}}
#endif
#else
_Static_assert(MB_EQ_WC, "!__STDC_MB_MIGHT_NEQ_WC__ but some character differs"); // expected-error {{C11}}
#endif
}

// dr631: na

namespace dr632 { // dr632: yes
  struct S { int n; } s = {{5}}; // expected-warning {{braces}}
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
  int k = f(S()); // expected-error {{cannot pass}}
}

namespace dr635 { // dr635: yes
  template<typename T> struct A { A(); ~A(); };
  template<typename T> A<T>::A<T>() {} // expected-error {{cannot have template arguments}}
  template<typename T> A<T>::~A<T>() {}

  template<typename T> struct B { B(); ~B(); };
  template<typename T> B<T>::B() {}
  template<typename T> B<T>::~B() {}

  struct C { template<typename T> C(); C(); };
  template<typename T> C::C() {}
  C::C() {}
  template<> C::C<int>() {} // expected-error {{constructor name}} expected-error {{unqualified-id}}
  /*FIXME: needed for error recovery:*/;

  template<typename T> struct D { template<typename U> D(); D(); };
  template<typename T> D<T>::D() {} // expected-note {{previous}}
  template<typename T> template<typename U> D<T>::D() {}
  template<typename T> D<T>::D<T>() {} // expected-error {{redefinition}} expected-error {{cannot have template arg}}
}

namespace dr637 { // dr637: yes
  void f(int i) {
    i = ++i + 1;
    i = i++ + 1; // expected-warning {{unsequenced}}
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
    template<class T> friend struct A<T>::B; // expected-warning {{not supported}}
    template<class T> friend void A<T>::f(); // expected-warning {{not supported}}
    template<class T> friend void A<T>::g(); // expected-warning {{not supported}}
    template<class T> friend void A<T>::C::h(); // expected-warning {{not supported}}
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

namespace dr639 { // dr639: yes
  void f(int i) {
    void((i = 0) + (i = 0)); // expected-warning {{unsequenced}}
  }
}
