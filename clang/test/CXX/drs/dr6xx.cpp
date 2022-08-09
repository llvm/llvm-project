// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking
// RUN: %clang_cc1 -std=c++20 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking

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
  template<typename T> void f(S<T> &&); // expected-note {{expects an rvalue}}
  template<typename T> void g(T &&);
  template<typename T> void h(const T &&); // expected-note {{expects an rvalue}}

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
    i = i++ + 1;
#if __cplusplus < 201703L
    // expected-warning@-2 {{unsequenced}}
#endif
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

namespace dr641 { // dr641: yes
  namespace std_example {
    struct abc;

    struct xyz {
      xyz(); // expected-note 0-1{{candidate}}
      xyz(xyz &); // expected-note 0-1{{candidate}}

      operator xyz &() = delete; // expected-error 0-1{{extension}} expected-warning {{will never be used}}
      operator abc &() = delete; // expected-error 0-1{{extension}}
    };

    struct abc : xyz {};

    template<typename T>
    void use(T &); // expected-note {{expects an lvalue}}
    void test() {
      use<xyz>(xyz()); // expected-error {{no match}}
      use<const xyz>(xyz());
#if __cplusplus < 201103L
      // expected-error-re@-2 {{no viable constructor copying parameter of type '{{.*}}xyz'}}
#endif
    }
  }

  template<typename T> struct error { typedef typename T::error type; };

  struct A {
    template<typename T, typename error<T>::type = 0> operator T() const; // expected-error 0-1{{extension}}
  };
  A a;
  void f(A&); // expected-note 2{{candidate}}
  void g(const A ca) {
    f(A()); // expected-error {{no match}}
    f(ca); // expected-error {{no match}}
    (void)A();
    (void)ca;
  }
}

namespace dr642 { // dr642: yes
  void f() {
    const int i = 2;
    {
      char i[i];
      _Static_assert(sizeof(i) == 2, ""); // expected-error {{C11}}
    }
  }

  struct s { int a; };
  void g(int s) {
    struct s *p = new struct s;
    p->a = s;
  }
}

#if __cplusplus >= 201103L
namespace dr643 { // dr643: yes
  struct A {
    int x;
    auto f() -> decltype(this->x);
    auto f(A &a) -> decltype(a.x);
    auto g() -> decltype(x);
    auto h() -> decltype(this->y); // expected-error {{no member named 'y'}}
    auto h(A &a) -> decltype(a.y); // expected-error {{no member named 'y'}}
    auto i() -> decltype(y); // expected-error {{undeclared identifier 'y'}}
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
namespace dr647 { // dr647: yes
  // This is partially superseded by dr1358.
  struct A {
    constexpr virtual void f() const;
    constexpr virtual void g() const {}
#if __cplusplus <= 201703L
    // expected-error@-2 {{virtual function cannot be constexpr}}
#endif
  };

  struct X { virtual void f() const; };
#if __cplusplus <= 201703L
  // expected-note@-2 {{overridden}}
#endif
  struct B : X {
    constexpr void f() const {}
#if __cplusplus <= 201703L
    // expected-error@-2 {{virtual function cannot be constexpr}}
#endif
  };

  struct NonLiteral { NonLiteral() {} }; // expected-note {{not an aggregate and has no constexpr constructors}}

  struct C {
    constexpr C(NonLiteral);
    constexpr C(NonLiteral, int) {} // expected-error {{not a literal type}}
    constexpr C() try {} catch (...) {}
#if __cplusplus <= 201703L
    // expected-error@-2 {{function try block in constexpr constructor is a C++20 extension}}
#endif
#if __cplusplus < 201402L
    // expected-error@-5 {{use of this statement in a constexpr constructor is a C++14 extension}}
#endif
  };

  struct D {
    operator int() const;
    constexpr D(int) {}
    D(float); // expected-note 2{{declared here}}
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

    constexpr E(int) // expected-error {{never produces a constant expression}}
        : n(0),
          d(0.0f) {} // expected-note {{non-constexpr constructor}}
    constexpr E(float f) // expected-error {{never produces a constant expression}}
        : n(get()),
          d(D(0) + f) {} // expected-note {{non-constexpr constructor}}
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
namespace dr649 { // dr649: yes
alignas(0x200000000) int n;       // expected-error {{requested alignment}}1
struct alignas(0x200000000) X {}; // expected-error {{requested alignment}}
struct Y {
  int n alignas(0x200000000); // expected-error {{requested alignment}}
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
    if (nullptr) {} // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
    bool b = nullptr; // expected-error {{cannot initialize a variable of type 'bool' with an rvalue of type 'std::nullptr_t'}}
    if (nullptr == 0) {}
    if (nullptr != 0) {}
    if (nullptr <= 0) {} // expected-error {{invalid operands}}
    if (nullptr == 1) {} // expected-error {{invalid operands}}
    if (!nullptr) {} // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
    decltype(nullptr) n = 0;
    static_cast<int>(nullptr); // expected-error {{not allowed}}
    (void)static_cast<decltype(nullptr)>(0);
    static_cast<decltype(nullptr)>(1); // expected-error {{not allowed}}
    void(true ? nullptr : 0);
    void(true ? 0 : nullptr);
  }
}
#endif

namespace dr655 { // dr655: yes
  struct A { A(int); }; // expected-note 2-3{{not viable}}
                        // expected-note@-1 {{'dr655::A' declared here}}
  struct B : A {
    A a; // expected-note {{member is declared here}}
    B();
    B(int) : B() {} // expected-error 0-1 {{C++11}}
    B(int*) : A() {} // expected-error {{no matching constructor}}
                     // expected-error@-1 {{must explicitly initialize the member 'a'}}
  };
}

namespace dr656 { // dr656: yes
  struct A { A(const A&) = delete; }; // expected-error 0-1 {{C++11}}
  struct B : A {};
  struct X { operator B(); } x;
  const A &r = x;
  struct Y : private A { // expected-note 2{{here}} expected-note 2{{candidate}}
    operator B() volatile;
  };
  extern Y y;
  extern volatile Y vy;
  // Conversion not considered due to reference-related types.
  const A &s = y; // expected-error {{private base class}}
  const A &t = vy; // expected-error {{drops 'volatile'}}

  struct C { operator struct D(); } c;
  struct D : C {};
  const D &d = c; // ok, D not reference-related to C

  template<typename T> void accept(T); // expected-note {{candidate}}
  template<typename T> void accept(...) = delete; // expected-error 0-1 {{C++11}} expected-note {{candidate}}
  void f() {
    accept<const A&>(x);
    accept<const A&>(y); // expected-error {{private base class}}
    accept<const A&>(vy); // expected-error {{call to deleted}} expected-error {{no matching constructor}}
    accept<const D&>(c);
  }
}

namespace dr657 { // dr657: partial
  struct Abs { virtual void x() = 0; };
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

  // FIXME: We should reject this.
  struct C { C(Abs) {} };
  // FIXME: We should reject this.
  struct Q { operator Abs() { __builtin_unreachable(); } } q;
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
namespace dr659 { // dr659: yes
  static_assert(alignof(char) == alignof(char&), "");
  static_assert(alignof(int) == alignof(int&), "");
  int n = alignof(int(&)()); // expected-error {{application of 'alignof' to a function type}}
  struct A; // expected-note {{forward}}
  int m = alignof(A&); // expected-error {{application of 'alignof' to an incomplete type}}
}
#endif

#if __cplusplus >= 201103L
namespace dr660 { // dr660: yes
  enum : int { a };
  enum class { b }; // expected-error {{requires a name}}
  auto x = a;

  struct X {
    enum : int { a };
    enum class { b }; // expected-error {{requires a name}}
  };
  auto y = X::a;
}
#endif

// dr661 FIXME: add codegen test

namespace dr662 { // dr662: yes
  template <typename T> void f(T t) {
    T &tr = t;
    T *tp = &t; // expected-error {{pointer to a reference}}
#if __cplusplus >= 201103L
    auto *ap = &t;
#endif
  }
  void g(int n) { f<int&>(n); } // expected-note {{instantiation of}}
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

namespace dr665 { // dr665: yes
  struct A { virtual ~A(); };
  struct B : A {} *b;
  struct C : private A {} *c; // expected-note {{here}}
  struct D : B, C {} *d;

  struct VB : virtual A {} *vb;
  struct VC : private virtual A {} *vc; // expected-note {{here}}
  struct VD : VB, VC {} *vd;

  void f() {
    (void)dynamic_cast<A*>(b);
    (void)dynamic_cast<A*>(c); // expected-error {{private}}
    (void)dynamic_cast<A*>(d); // expected-error {{ambiguous}}
    (void)dynamic_cast<A*>(vb);
    (void)dynamic_cast<A*>(vc); // expected-error {{private}}, even though it could be valid at runtime
    (void)dynamic_cast<A*>(vd);
  }
}

namespace dr666 { // dr666: yes
  struct P { friend P operator*(P, P); P(int); } p(0);

  template<int> int f();
  template<typename T> int f() {
    T::type *p = 0; // expected-error {{missing 'typename'}}
    int a(T::type); // expected-error {{missing 'typename'}}
    return f<T::type>(); // expected-error {{missing 'typename'}}
  }
  struct X { static const int type = 0; };
  struct Y { typedef int type; };
  int a = f<X>();
  int b = f<Y>(); // expected-note {{instantiation of}}
}

// Triviality is entirely different in C++98.
#if __cplusplus >= 201103L
namespace dr667 { // dr667: yes
  struct A {
    A() = default; // expected-warning {{explicitly defaulted default constructor is implicitly deleted}}
    int &r; // expected-note {{because field 'r' of reference type 'int &' would not be initialized}}
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

namespace dr671 { // dr671: yes
  enum class E { e }; // expected-error 0-1 {{C++11}}
  E e = static_cast<E>(0);
  int n = static_cast<int>(E::e); // expected-error 0-1 {{C++11}}
  int m = static_cast<int>(e); // expected-error 0-1 {{C++11}}
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
  F *f; // expected-error {{unknown type name}}
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
    int n; // expected-note 2{{private}}
  };

  template<typename T> int f(T) { return X().n; }
  int g(int) { return X().n; }
  template<typename T> int g(T) { return X().n; } // expected-error {{private}}
  int h(int) { return X().n; } // expected-error {{private}}
  template<typename T> int h(T) { return X().n; }

  template int f(int);
  template int g(int); // expected-note {{in instantiation of}}
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
    int n; // expected-note 2{{private}}
  };

  template<typename T> int Y::f(T) { return Z().n; }
  int Y::g(int) { return Z().n; }
  template<typename T> int Y::g(T) { return Z().n; } // expected-error {{private}}
  int Y::h(int) { return Z().n; } // expected-error {{private}}
  template<typename T> int Y::h(T) { return Z().n; }

  // FIXME: Should the <> be required here?
  template int Y::f<>(int);
  template int Y::g<>(int); // expected-note {{in instantiation of}}
  template int Y::h<>(int);
}

namespace dr675 { // dr675: dup 739
  template<typename T> struct A { T n : 1; };
#if __cplusplus >= 201103L
  static_assert(A<char>{1}.n < 0, ""); // expected-warning {{implicit truncation from 'int' to bit-field changes value from 1 to -1}}
  static_assert(A<int>{1}.n < 0, ""); // expected-warning {{implicit truncation from 'int' to bit-field changes value from 1 to -1}}
  static_assert(A<long long>{1}.n < 0, ""); // expected-warning {{implicit truncation from 'int' to bit-field changes value from 1 to -1}}
#endif
}

// dr676: na

namespace dr677 { // dr677: no
  struct A {
    void *operator new(std::size_t);
    void operator delete(void*) = delete; // expected-error 0-1{{C++11}} expected-note {{deleted}}
  };
  struct B {
    void *operator new(std::size_t);
    void operator delete(void*) = delete; // expected-error 0-1{{C++11}} expected-note 2{{deleted}}
    virtual ~B();
  };
  void f(A *p) { delete p; } // expected-error {{deleted}}
  // FIXME: This appears to be valid; we shouldn't even be looking up the 'operator delete' here.
  void f(B *p) { delete p; } // expected-error {{deleted}}
  B::~B() {} // expected-error {{deleted}}
}

// dr678 FIXME: check that the modules ODR check catches this

namespace dr679 { // dr679: yes
  struct X {};
  template<int> void operator+(X, X);
  template<> void operator+<0>(X, X) {} // expected-note {{previous}}
  template<> void operator+<0>(X, X) {} // expected-error {{redefinition}}
}

// dr680: na

#if __cplusplus >= 201103L
namespace dr681 { // dr681: partial
  auto *a() -> int; // expected-error {{must specify return type 'auto', not 'auto *'}}
  auto (*b)() -> int;
  // FIXME: The errors here aren't great.
  auto (*c()) -> int; // expected-error {{expected function body}}
  auto ((*d)()) -> int; // expected-error {{expected ';'}} expected-error {{requires an initializer}}

  // FIXME: This is definitely wrong. This should be
  //   "function of () returning pointer to function of () returning int"
  // not a function with a deduced return type.
  auto (*e())() -> int; // expected-error 0-1{{C++14}}

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
    int a; // expected-note {{here}}
    constexpr int *p = &a; // expected-error {{constant expression}} expected-note {{pointer to 'a'}}
  }
}
#endif

namespace dr685 { // dr685: yes
  enum E : long { e };
#if __cplusplus < 201103L
    // expected-error@-2 {{enumeration types with a fixed underlying type are a C++11 extension}}
#endif
  void f(int);
  int f(long);
  int a = f(e);

  enum G : short { g };
#if __cplusplus < 201103L
    // expected-error@-2 {{enumeration types with a fixed underlying type are a C++11 extension}}
#endif
  int h(short);
  void h(long);
  int b = h(g);

  int i(int);
  void i(long);
  int c = i(g);

  int j(unsigned int); // expected-note {{candidate}}
  void j(long); // expected-note {{candidate}}
  int d = j(g); // expected-error {{ambiguous}}

  // Valid per dr1601
  int k(short);
  void k(int);
  int x = k(g);
}

namespace dr686 { // dr686: yes
  void f() {
    (void)dynamic_cast<struct A*>(0); // expected-error {{incomplete}} expected-note {{forward}}
    (void)dynamic_cast<struct A{}*>(0); // expected-error {{cannot be defined in a type specifier}}
    (void)typeid(struct B*);
    (void)typeid(struct B{}*); // expected-error {{cannot be defined in a type specifier}}
    (void)static_cast<struct C*>(0);
    (void)static_cast<struct C{}*>(0); // expected-error {{cannot be defined in a type specifier}}
    (void)reinterpret_cast<struct D*>(0);
    (void)reinterpret_cast<struct D{}*>(0); // expected-error {{cannot be defined in a type specifier}}
    (void)const_cast<struct E*>(0); // expected-error {{not allowed}}
    (void)const_cast<struct E{}*>(0); // expected-error {{cannot be defined in a type specifier}}
    (void)sizeof(struct F*);
    (void)sizeof(struct F{}*); // expected-error {{cannot be defined in a type specifier}}
    (void)new struct G*;
    (void)new struct G{}*; // expected-error {{cannot be defined in a type specifier}}
#if __cplusplus >= 201103L
    (void)alignof(struct H*);
    (void)alignof(struct H{}*); // expected-error {{cannot be defined in a type specifier}}
#endif
    (void)(struct I*)0;
    (void)(struct I{}*)0; // expected-error {{cannot be defined in a type specifier}}
    if (struct J *p = 0) {}
    if (struct J {} *p = 0) {} // expected-error {{cannot be defined in a condition}}
    for (struct K *p = 0; struct L *q = 0; ) {}
    for (struct K {} *p = 0; struct L {} *q = 0; ) {} // expected-error {{'L' cannot be defined in a condition}}
#if __cplusplus >= 201103L
    using M = struct {};
#endif
    struct N {
      operator struct O{}(){}; // expected-error {{cannot be defined in a type specifier}}
    };
    try {}
    catch (struct P *) {} // expected-error {{incomplete}} expected-note {{forward}}
    catch (struct P {} *) {} // expected-error {{cannot be defined in a type specifier}}
#if __cplusplus < 201703L
    void g() throw(struct Q); // expected-error {{incomplete}} expected-note {{forward}}
    void h() throw(struct Q {}); // expected-error {{cannot be defined in a type specifier}}
#endif
  }
  template<struct R *> struct X;
  template<struct R {} *> struct Y; // expected-error {{cannot be defined in a type specifier}}
}

namespace dr687 { // dr687 (9 c++20, but the issue is still considered open)
  template<typename T> void f(T a) {
    // This is valid in C++20.
    g<int>(a);
#if __cplusplus <= 201703L
    // expected-error@-2 {{C++20 extension}}
#endif

    // This is not.
    template g<int>(a); // expected-error {{expected expression}}
  }
}

namespace dr692 { // dr692: no
  namespace temp_func_order_example2 {
    template <typename T, typename U> struct A {};
    template <typename T, typename U> void f(U, A<U, T> *p = 0); // expected-note {{candidate}}
    template <typename U> int &f(U, A<U, U> *p = 0); // expected-note {{candidate}}
    template <typename T> void g(T, T = T());
    template <typename T, typename... U> void g(T, U...); // expected-error 0-1{{C++11}}
    void h() {
      int &r = f<int>(42, (A<int, int> *)0);
      f<int>(42); // expected-error {{ambiguous}}
      // FIXME: We should reject this due to ambiguity between the pack and the
      // default argument. Only parameters with arguments are considered during
      // partial ordering of function templates.
      g(42);
    }
  }

  namespace temp_func_order_example3 {
    template <typename T, typename... U> void f(T, U...); // expected-error 0-1{{C++11}}
    template <typename T> void f(T);
    template <typename T, typename... U> int &g(T *, U...); // expected-error 0-1{{C++11}}
    template <typename T> void g(T);
    void h(int i) {
      // This is made ambiguous by dr692, but made valid again by dr1395.
      f(&i);
      int &r = g(&i);
    }
  }

  namespace temp_deduct_partial_example {
    template <typename... Args> char &f(Args... args); // expected-error 0-1{{C++11}}
    template <typename T1, typename... Args> short &f(T1 a1, Args... args); // expected-error 0-1{{C++11}}
    template <typename T1, typename T2> int &f(T1 a1, T2 a2);
    void g() {
      char &a = f();
      short &b = f(1, 2, 3);
      int &c = f(1, 2);
    }
  }

  namespace temp_deduct_type_example1 {
    template <class T1, class ...Z> class S; // expected-error 0-1{{C++11}}
    template <class T1, class ...Z> class S<T1, const Z&...>; // expected-error 0-1{{C++11}}
    template <class T1, class T2> class S<T1, const T2&> {};
    S<int, const int&> s;

    // FIXME: This should select the first partial specialization. Deduction of
    // the second from the first should succeed, because we should ignore the
    // trailing pack in A with no corresponding P.
    template<class T, class... U> struct A; // expected-error 0-1{{C++11}}
    template<class T1, class T2, class... U> struct A<T1,T2*,U...>; // expected-note {{matches}} expected-error 0-1{{C++11}}
    template<class T1, class T2> struct A<T1,T2> {}; // expected-note {{matches}}
    template struct A<int, int*>; // expected-error {{ambiguous}}
  }

  namespace temp_deduct_type_example3 {
    // FIXME: This should select the first template, as in the case above.
    template<class T, class... U> void f(T*, U...){} // expected-note {{candidate}} expected-error 0-1{{C++11}}
    template<class T> void f(T){} // expected-note {{candidate}}
    template void f(int*); // expected-error {{ambiguous}}
  }
}

namespace dr696 { // dr696: yes
  void f(const int*);
  void g() {
    const int N = 10; // expected-note 1+{{here}}
    struct A {
      void h() {
        int arr[N]; (void)arr;
        f(&N); // expected-error {{declared in enclosing}}
      }
    };
#if __cplusplus >= 201103L
    (void) [] { int arr[N]; (void)arr; };
    (void)[] { f(&N); }; // expected-error {{cannot be implicitly captured}} expected-note {{here}} expected-note 2 {{capture 'N' by}} expected-note 2 {{default capture by}}
#endif
  }
}
