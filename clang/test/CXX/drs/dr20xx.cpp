// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors \
// RUN:            -Wno-variadic-macros -Wno-c11-extensions
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2a -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
#define static_assert(...) _Static_assert(__VA_ARGS__)
#endif

namespace dr2026 { // dr2026: 11
  template<int> struct X {};

  const int a = a + 1; // expected-warning {{uninitialized}} expected-note {{here}} expected-note 0-1{{outside its lifetime}}
  X<a> xa; // expected-error {{constant expression}} expected-note {{initializer of 'a'}}

#if __cplusplus >= 201103L
  constexpr int b = b; // expected-error {{constant expression}} expected-note {{outside its lifetime}}
  [[clang::require_constant_initialization]] int c = c; // expected-error {{constant initializer}} expected-note {{attribute}}
#if __cplusplus == 201103L
  // expected-note@-2 {{read of non-const variable}} expected-note@-2 {{declared here}}
#else
  // expected-note@-4 {{outside its lifetime}}
#endif
#endif

#if __cplusplus > 201703L
  constinit int d = d; // expected-error {{constant initializer}} expected-note {{outside its lifetime}} expected-note {{'constinit'}}
#endif

  void f() {
    static const int e = e + 1; // expected-warning {{suspicious}} expected-note {{here}} expected-note 0-1{{outside its lifetime}}
    X<e> xe; // expected-error {{constant expression}} expected-note {{initializer of 'e'}}

#if __cplusplus >= 201103L
    static constexpr int f = f; // expected-error {{constant expression}} expected-note {{outside its lifetime}}
    [[clang::require_constant_initialization]] static int g = g; // expected-error {{constant initializer}} expected-note {{attribute}}
#if __cplusplus == 201103L
    // expected-note@-2 {{read of non-const variable}} expected-note@-2 {{declared here}}
#else
    // expected-note@-4 {{outside its lifetime}}
#endif
#endif

#if __cplusplus > 201703L
    static constinit int h = h; // expected-error {{constant initializer}} expected-note {{outside its lifetime}} expected-note {{'constinit'}}
#endif
  }
}

namespace dr2082 { // dr2082: 11
  void test1(int x, int = sizeof(x)); // ok
#if __cplusplus >= 201103L
  void test2(int x, int = decltype(x){}); // ok
#endif
}

namespace dr2083 { // dr2083: partial
#if __cplusplus >= 201103L
  void non_const_mem_ptr() {
    struct A {
      int x;
      int y;
    };
    constexpr A a = {1, 2};
    struct B {
      int A::*p;
      constexpr int g() const {
        // OK, not an odr-use of 'a'.
        return a.*p;
      };
    };
    static_assert(B{&A::x}.g() == 1, "");
    static_assert(B{&A::y}.g() == 2, "");
  }
#endif

  const int a = 1;
  int b;
  // Note, references only get special odr-use / constant initializxer
  // treatment in C++11 onwards. We continue to apply that even after DR2083.
  void ref_to_non_const() {
    int c;
    const int &ra = a; // expected-note 0-1{{here}}
    int &rb = b; // expected-note 0-1{{here}}
    int &rc = c; // expected-note {{here}}
    struct A {
      int f() {
        int a = ra;
        int b = rb;
#if __cplusplus < 201103L
        // expected-error@-3 {{in enclosing function}}
        // expected-error@-3 {{in enclosing function}}
#endif
        int c = rc; // expected-error {{in enclosing function}}
        return a + b + c;
      }
    };
  }

#if __cplusplus >= 201103L
  struct NoMut1 { int a, b; };
  struct NoMut2 { NoMut1 m; };
  struct NoMut3 : NoMut1 {
    constexpr NoMut3(int a, int b) : NoMut1{a, b} {}
  };
  struct Mut1 {
    int a;
    mutable int b;
  };
  struct Mut2 { Mut1 m; };
  struct Mut3 : Mut1 {
    constexpr Mut3(int a, int b) : Mut1{a, b} {}
  };
  void mutable_subobjects() {
    constexpr NoMut1 nm1 = {1, 2};
    constexpr NoMut2 nm2 = {1, 2};
    constexpr NoMut3 nm3 = {1, 2};
    constexpr Mut1 m1 = {1, 2}; // expected-note {{declared here}}
    constexpr Mut2 m2 = {1, 2}; // expected-note {{declared here}}
    constexpr Mut3 m3 = {1, 2}; // expected-note {{declared here}}
    struct A {
      void f() {
        static_assert(nm1.a == 1, "");
        static_assert(nm2.m.a == 1, "");
        static_assert(nm3.a == 1, "");
        // Can't even access a non-mutable member of a variable containing mutable fields.
        static_assert(m1.a == 1, ""); // expected-error {{enclosing function}}
        static_assert(m2.m.a == 1, ""); // expected-error {{enclosing function}}
        static_assert(m3.a == 1, ""); // expected-error {{enclosing function}}
      }
    };
  }
#endif

  void ellipsis() {
    void ellipsis(...);
    struct A {};
    const int n = 0;
#if __cplusplus >= 201103L
    constexpr
#endif
      A a = {}; // expected-note {{here}}
    struct B {
      void f() {
        ellipsis(n);
        // Even though this is technically modelled as an lvalue-to-rvalue
        // conversion, it calls a constructor and binds 'a' to a reference, so
        // it results in an odr-use.
        ellipsis(a); // expected-error {{enclosing function}}
      }
    };
  }

#if __cplusplus >= 201103L
  void volatile_lval() {
    struct A { int n; };
    constexpr A a = {0}; // expected-note {{here}}
    struct B {
      void f() {
        // An lvalue-to-rvalue conversion of a volatile lvalue always results
        // in odr-use.
        int A::*p = &A::n;
        int x = a.*p;
        volatile int A::*q = p;
        int y = a.*q; // expected-error {{enclosing function}}
      }
    };
  }
#endif

  void discarded_lval() {
    struct A { int x; mutable int y; volatile int z; };
    A a; // expected-note 1+{{here}}
    int &r = a.x; // expected-note {{here}}
    struct B {
      void f() {
        a.x; // expected-warning {{unused}}
        a.*&A::x; // expected-warning {{unused}}
        true ? a.x : a.y; // expected-warning {{unused}}
        (void)a.x;
        a.x, discarded_lval(); // expected-warning {{unused}}
#if 1 // FIXME: These errors are all incorrect; the above code is valid.
      // expected-error@-6 {{enclosing function}}
      // expected-error@-6 {{enclosing function}}
      // expected-error@-6 2{{enclosing function}}
      // expected-error@-6 {{enclosing function}}
      // expected-error@-6 {{enclosing function}}
#endif

        // 'volatile' qualifier triggers an lvalue-to-rvalue conversion.
        a.z; // expected-error {{enclosing function}}
#if __cplusplus < 201103L
        // expected-warning@-2 {{assign into a variable}}
#endif

        // References always get "loaded" to determine what they reference,
        // even if the result is discarded.
        r; // expected-error {{enclosing function}} expected-warning {{unused}}
      }
    };
  }

  namespace dr_example_1 {
    extern int globx;
    int main() {
      const int &x = globx;
      struct A {
#if __cplusplus < 201103L
        // expected-error@+2 {{enclosing function}} expected-note@-3 {{here}}
#endif
        const int *foo() { return &x; }
      } a;
      return *a.foo();
    }
  }

#if __cplusplus >= 201103L
  namespace dr_example_2 {
    struct A {
      int q;
      constexpr A(int q) : q(q) {}
      constexpr A(const A &a) : q(a.q * 2) {} // (note, not called)
    };

    int main(void) {
      constexpr A a(42);
      constexpr int aq = a.q;
      struct Q {
        int foo() { return a.q; }
      } q;
      return q.foo();
    }

    // Checking odr-use does not invent an lvalue-to-rvalue conversion (and
    // hence copy construction) on the potential result variable.
    struct B {
      int b = 42;
      constexpr B() {}
      constexpr B(const B&) = delete;
    };
    void f() {
      constexpr B b;
      struct Q {
        constexpr int foo() const { return b.b; }
      };
      static_assert(Q().foo() == 42, "");
    }
  }
#endif
}

namespace dr2094 { // dr2094: 5
  struct A { int n; };
  struct B { volatile int n; };
  static_assert(__is_trivially_copyable(volatile int), "");
  static_assert(__is_trivially_copyable(const volatile int), "");
  static_assert(__is_trivially_copyable(const volatile int[]), "");
  static_assert(__is_trivially_copyable(A), "");
  static_assert(__is_trivially_copyable(volatile A), "");
  static_assert(__is_trivially_copyable(const volatile A), "");
  static_assert(__is_trivially_copyable(const volatile A[]), "");
  static_assert(__is_trivially_copyable(B), "");

  static_assert(__is_trivially_constructible(A, A const&), "");
  static_assert(__is_trivially_constructible(B, B const&), "");

  static_assert(__is_trivially_assignable(A, const A&), "");
  static_assert(__is_trivially_assignable(B, const B&), "");
}
