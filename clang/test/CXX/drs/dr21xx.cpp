// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-error@+1 {{variadic macro}}
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
#endif

namespace dr2100 { // dr2100: 12
  template<const int *P, bool = true> struct X {};
  template<typename T> struct A {
    static const int n = 1;
    int f() {
      return X<&n>::n; // ok, value-dependent
    }
    int g() {
      static const int n = 2;
      return X<&n>::n; // ok, value-dependent
#if __cplusplus < 201702L
      // expected-error@-2 {{does not have linkage}} expected-note@-3 {{here}}
#endif
    }
  };
  template<const int *P> struct X<P> {
#if __cplusplus < 201103L
    static const int n = 0;
#else
    static const int n = *P;
#endif
  };
  int q = A<int>().f() + A<int>().g();

  // Corresponding constructs where the address is not taken are not
  // value-dependent.
  template<int N, bool = true> struct Y {};
  template<typename T> struct B {
    static const int n = 1;
    int f() {
      return Y<n>::declared_later; // expected-error {{no member named 'declared_later'}}
    }
    int g() {
      static const int n = 2;
      return Y<n>::declared_later; // expected-error {{no member named 'declared_later'}}
    }
  };
  template<int N> struct Y<N> {
    static const int declared_later = 0;
  };
}

namespace dr2103 { // dr2103: yes
  void f() {
    int a;
    int &r = a; // expected-note {{here}}
    struct Inner {
      void f() {
        int &s = r; // expected-error {{enclosing function}}
        (void)s;
      }
    };
  }
}

namespace dr2120 { // dr2120: 7
  struct A {};
  struct B : A {};
  struct C { A a; };
  struct D { C c[5]; };
  struct E : B { D d; };
  static_assert(__is_standard_layout(B), "");
  static_assert(__is_standard_layout(D), "");
  static_assert(!__is_standard_layout(E), "");
}

namespace dr2126 { // dr2126: 12
#if __cplusplus >= 201103L
  struct A { int n; };

  const A &a = {1};              // const temporary
  A &b = (A &)(const A &)A{1};   // const temporary
  A &&c = (A &&)(const A &)A{1}; // const temporary

  A &&d = {1};                   // non-const temporary expected-note {{here}}
  const A &e = (A &)(A &&) A{1}; // non-const temporary expected-note {{here}}
  A &&f = (A &&)(A &&) A{1};     // non-const temporary expected-note {{here}}

  constexpr const A &g = {1};    // const temporary
  constexpr A &&h = {1};         // non-const temporary expected-note {{here}}

  struct B { const A &a; };
  B i = {{1}};           // extending decl not usable in constant expr expected-note {{here}}
  const B j = {{1}};     // extending decl not usable in constant expr expected-note {{here}}
  constexpr B k = {{1}}; // extending decl usable in constant expr

  static_assert(a.n == 1, "");
  static_assert(b.n == 1, "");
  static_assert(c.n == 1, "");
  static_assert(d.n == 1, ""); // expected-error {{constant}} expected-note {{read of temporary}}
  static_assert(e.n == 1, ""); // expected-error {{constant}} expected-note {{read of temporary}}
  static_assert(f.n == 1, ""); // expected-error {{constant}} expected-note {{read of temporary}}
  static_assert(g.n == 1, "");
  static_assert(h.n == 1, ""); // expected-error {{constant}} expected-note {{read of temporary}}
  static_assert(i.a.n == 1, ""); // expected-error {{constant}} expected-note {{read of non-constexpr variable}}
  static_assert(j.a.n == 1, ""); // expected-error {{constant}} expected-note {{read of temporary}}
  static_assert(k.a.n == 1, "");
#endif
}

namespace dr2140 { // dr2140: 9
#if __cplusplus >= 201103L
  union U { int a; decltype(nullptr) b; };
  constexpr int *test(U u) {
    return u.b;
  }
  static_assert(!test({123}), "u.b should be valid even when b is inactive");
#endif
}

namespace dr2157 { // dr2157: 11
#if __cplusplus >= 201103L
  enum E : int;
  struct X {
    enum dr2157::E : int(); // expected-error {{only allows ':' in member enumeration declaration to introduce a fixed underlying type}}
  };
#endif
}

namespace dr2170 { // dr2170: 9
#if __cplusplus >= 201103L
  void f() {
    constexpr int arr[3] = {1, 2, 3}; // expected-note {{here}}
    struct S {
      int get(int n) { return arr[n]; }
      const int &get_ref(int n) { return arr[n]; } // expected-error {{enclosing function}}
      // FIXME: expected-warning@-1 {{reference to stack}}
    };
  }
#endif
}

namespace dr2171 { // dr2171: 15
#if __cplusplus >= 201103L

struct NonConstCopy {
  NonConstCopy(NonConstCopy &) = default;
  NonConstCopy &operator=(NonConstCopy &) = default;
};

static_assert(__has_trivial_copy(NonConstCopy), "");
static_assert(__is_trivially_constructible(NonConstCopy, NonConstCopy &), "");
static_assert(!__is_trivially_constructible(NonConstCopy, NonConstCopy), "");
static_assert(!__is_trivially_constructible(NonConstCopy, const NonConstCopy &), "");
static_assert(!__is_trivially_constructible(NonConstCopy, NonConstCopy &&), "");

static_assert(__has_trivial_assign(NonConstCopy), "");
static_assert(__is_trivially_assignable(NonConstCopy &, NonConstCopy &), "");
static_assert(!__is_trivially_assignable(NonConstCopy &, const NonConstCopy &), "");
static_assert(!__is_trivially_assignable(NonConstCopy &, NonConstCopy), "");
static_assert(!__is_trivially_assignable(NonConstCopy &, NonConstCopy &&), "");
static_assert(__is_trivially_assignable(NonConstCopy &&, NonConstCopy &), "");
static_assert(!__is_trivially_assignable(NonConstCopy &&, const NonConstCopy &), "");
static_assert(!__is_trivially_assignable(NonConstCopy &&, NonConstCopy), "");
static_assert(!__is_trivially_assignable(NonConstCopy &&, NonConstCopy &&), "");

#endif
} // namespace dr2171

namespace dr2180 { // dr2180: yes
  class A {
    A &operator=(const A &); // expected-note 0-2{{here}}
    A &operator=(A &&); // expected-note 0-2{{here}} expected-error 0-1{{extension}}
  };

  struct B : virtual A {
    B &operator=(const B &);
    B &operator=(B &&); // expected-error 0-1{{extension}}
    virtual void foo() = 0;
  };
#if __cplusplus < 201103L
  B &B::operator=(const B&) = default; // expected-error {{private member}} expected-error {{extension}} expected-note {{here}}
  B &B::operator=(B&&) = default; // expected-error {{private member}} expected-error 2{{extension}} expected-note {{here}}
#else
  B &B::operator=(const B&) = default; // expected-error {{would delete}} expected-note@-9{{inaccessible copy assignment}}
  B &B::operator=(B&&) = default; // expected-error {{would delete}} expected-note@-10{{inaccessible move assignment}}
#endif
}
