// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx14,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx14,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx14,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

namespace cwg2007 { // cwg2007: 3.4
template<typename T> struct A { typename T::error e; };
template<typename T> struct B { };
B<A<void> > b1;
B<A<void> > b2 = b1;
int a = b2[0];
// cxx98-error@-1 {{type 'B<A<void> >' does not provide a subscript operator}}
// since-cxx11-error@-2 {{type 'B<A<void>>' does not provide a subscript operator}}
int b = __builtin_addressof(b2)->foo;
// cxx98-error@-1 {{no member named 'foo' in 'cwg2007::B<cwg2007::A<void> >'}}
// since-cxx11-error@-2 {{no member named 'foo' in 'cwg2007::B<cwg2007::A<void>>'}}
} // namespace cwg2007

// cwg2009: na

namespace cwg2026 { // cwg2026: 11
  template<int> struct X {};

  const int a = a + 1; // #cwg2026-a
  // expected-warning@-1 {{variable 'a' is uninitialized when used within its own initialization}}
  X<a> xa; // #cwg2026-xa
  // cxx98-error@-1 {{non-type template argument of type 'int' is not an integral constant expression}}
  //   cxx98-note@-2 {{initializer of 'a' is not a constant expression}}
  //   cxx98-note@#cwg2026-a {{declared here}}
  // since-cxx11-error@#cwg2026-xa {{non-type template argument is not a constant expression}}
  //   since-cxx11-note@#cwg2026-xa {{initializer of 'a' is not a constant expression}}
  //   since-cxx11-note@#cwg2026-a {{declared here}}

#if __cplusplus >= 201103L
  constexpr int b = b;
  // since-cxx11-error@-1 {{constexpr variable 'b' must be initialized by a constant expression}}
  //   since-cxx11-note@-2 {{read of object outside its lifetime is not allowed in a constant expression}}
  [[clang::require_constant_initialization]] int c = c;
  // since-cxx11-error@-1 {{variable does not have a constant initializer}}
  //   since-cxx11-note@-2 {{required by 'require_constant_initialization' attribute here}}
  //   cxx11-note@-3 {{read of non-const variable 'c' is not allowed in a constant expression}}
  //   cxx11-note@-4 {{declared here}}
  //   since-cxx14-note@-5 {{read of object outside its lifetime is not allowed in a constant expression}}
#endif

#if __cplusplus >= 202002L
  constinit int d = d;
  // since-cxx20-error@-1 {{variable does not have a constant initializer}}
  //   since-cxx20-note@-2 {{required by 'constinit' specifier here}}
  //   since-cxx20-note@-3 {{read of object outside its lifetime is not allowed in a constant expression}}
#endif

  void f() {
    static const int e = e + 1; // #cwg2026-e
    // expected-warning@-1 {{static variable 'e' is suspiciously used within its own initialization}}
    X<e> xe; // #cwg2026-xe
    // cxx98-error@-1 {{non-type template argument of type 'int' is not an integral constant expression}}
    //   cxx98-note@-2 {{initializer of 'e' is not a constant expression}}
    //   cxx98-note@#cwg2026-e {{declared here}}
    // since-cxx11-error@#cwg2026-xe {{non-type template argument is not a constant expression}}
    //   since-cxx11-note@#cwg2026-xe {{initializer of 'e' is not a constant expression}}
    //   since-cxx11-note@#cwg2026-e {{declared here}}

#if __cplusplus >= 201103L
    static constexpr int f = f;
    // since-cxx11-error@-1 {{constexpr variable 'f' must be initialized by a constant expression}}
    //   since-cxx11-note@-2 {{read of object outside its lifetime is not allowed in a constant expression}}
    [[clang::require_constant_initialization]] static int g = g;
    // since-cxx11-error@-1 {{variable does not have a constant initializer}}
    //   since-cxx11-note@-2 {{required by 'require_constant_initialization' attribute here}}
    //   cxx11-note@-3 {{read of non-const variable 'g' is not allowed in a constant expression}}
    //   cxx11-note@-4 {{declared here}}
    //   since-cxx14-note@-5 {{read of object outside its lifetime is not allowed in a constant expression}}
#endif

#if __cplusplus >= 202002L
    static constinit int h = h;
    // since-cxx20-error@-1 {{variable does not have a constant initializer}}
    //   since-cxx20-note@-2 {{required by 'constinit' specifier here}}
    //   since-cxx20-note@-3 {{read of object outside its lifetime is not allowed in a constant expression}}
#endif
  }
} // namespace cwg2026

namespace cwg2049 { // cwg2049: 18
#if __cplusplus >= 202302L
template <int* x = {}> struct X {};
X<> a;
X<nullptr> b;
static_assert(__is_same(decltype(a), decltype(b)));
#endif
} // namespace cwg2049

namespace cwg2061 { // cwg2061: 2.7
#if __cplusplus >= 201103L
  namespace A {
    inline namespace b {
      namespace C {
        // 'f' is the example from the DR.  'S' is an example where if we didn't
        // properly handle the two being the same, we would get an incomplete
        // type error during attempted instantiation.
        template<typename T> void f();
        template<typename T> struct S;
      }
    }
  }

  namespace A {
    namespace C {
      template<> void f<int>() { }
      template<> struct S<int> { };
    }
  }

  void use() {
    A::C::f<int>();
    A::C::S<int> s;
  }
#endif // C++11
} // namespace cwg2061

namespace cwg2076 { // cwg2076: 13
#if __cplusplus >= 201103L
  namespace std_example {
    struct A { A(int); };
    struct B { B(A); };
    B b{{0}};

    struct Params { int a; int b; };
    struct Foo {
      Foo(Params);
    };
    Foo foo{{1, 2}};
  }

  struct string_view {
    string_view(int); // not an aggregate
  };
  struct string {
    string(int); // not an aggregate
    operator string_view() const;
  };

  void foo(const string &); // #cwg2076-foo 
  void bar(string_view); // #cwg2076-bar

  void func(const string &arg) {
    // An argument in one set of braces is subject to user-defined conversions;
    // an argument in two sets of braces is not, but an identity conversion is
    // still OK.
    foo(arg);
    foo({arg});
    foo({{arg}});
    foo({{{arg}}});
    // since-cxx11-error@-1 {{no matching function}}
    //   since-cxx11-note@#cwg2076-foo  {{cannot convert initializer list}}
    bar(arg);
    bar({arg});
    bar({{arg}});
    // since-cxx11-error@-1 {{no matching function}}
    //   since-cxx11-note@#cwg2076-bar {{cannot convert initializer list}}
    bar({{{arg}}});
    // since-cxx11-error@-1 {{no matching function}}
    //   since-cxx11-note@#cwg2076-bar {{cannot convert initializer list}}
  }
#endif
} // namespace cwg2076

namespace cwg2082 { // cwg2082: 11
  void test1(int x, int = sizeof(x)); // ok
#if __cplusplus >= 201103L
  void test2(int x, int = decltype(x){}); // ok
#endif
} // namespace cwg2082

namespace cwg2083 { // cwg2083: partial
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
  // treatment in C++11 onwards. We continue to apply that even after CWG2083.
  void ref_to_non_const() {
    int c;
    const int &ra = a; // #cwg2083-ra
    int &rb = b; // #cwg2083-rb
    int &rc = c; // #cwg2083-rc
    struct A {
      int f() {
        int a = ra;
        // cxx98-error@-1 {{reference to local variable 'ra' declared in enclosing function 'cwg2083::ref_to_non_const'}}
        //   cxx98-note@#cwg2083-ra {{'ra' declared here}}
        int b = rb;
        // cxx98-error@-1 {{reference to local variable 'rb' declared in enclosing function 'cwg2083::ref_to_non_const'}}
        //   cxx98-note@#cwg2083-rb {{'rb' declared here}}
        int c = rc;
        // expected-error@-1 {{reference to local variable 'rc' declared in enclosing function 'cwg2083::ref_to_non_const'}}
        //   expected-note@#cwg2083-rc {{'rc' declared here}}
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
    constexpr Mut1 m1 = {1, 2}; // #cwg2083-m1
    constexpr Mut2 m2 = {1, 2}; // #cwg2083-m2
    constexpr Mut3 m3 = {1, 2}; // #cwg2083-m3
    struct A {
      void f() {
        static_assert(nm1.a == 1, "");
        static_assert(nm2.m.a == 1, "");
        static_assert(nm3.a == 1, "");
        // Can't even access a non-mutable member of a variable containing mutable fields.
        static_assert(m1.a == 1, "");
        // since-cxx11-error@-1 {{reference to local variable 'm1' declared in enclosing function 'cwg2083::mutable_subobjects'}}
        //   since-cxx11-note@#cwg2083-m1 {{'m1' declared here}}
        static_assert(m2.m.a == 1, "");
        // since-cxx11-error@-1 {{reference to local variable 'm2' declared in enclosing function 'cwg2083::mutable_subobjects'}}
        //   since-cxx11-note@#cwg2083-m2 {{'m2' declared here}}
        static_assert(m3.a == 1, "");
        // since-cxx11-error@-1 {{reference to local variable 'm3' declared in enclosing function 'cwg2083::mutable_subobjects'}}
        //   since-cxx11-note@#cwg2083-m3 {{'m3' declared here}}
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
      A a = {}; // #cwg2083-a
    struct B {
      void f() {
        ellipsis(n);
        // Even though this is technically modelled as an lvalue-to-rvalue
        // conversion, it calls a constructor and binds 'a' to a reference, so
        // it results in an odr-use.
        ellipsis(a);
        // expected-error@-1 {{reference to local variable 'a' declared in enclosing function 'cwg2083::ellipsis'}}
        //   expected-note@#cwg2083-a {{'a' declared here}}
      }
    };
  }

#if __cplusplus >= 201103L
  void volatile_lval() {
    struct A { int n; };
    constexpr A a = {0}; // #cwg2083-a2
    struct B {
      void f() {
        // An lvalue-to-rvalue conversion of a volatile lvalue always results
        // in odr-use.
        int A::*p = &A::n;
        int x = a.*p;
        volatile int A::*q = p;
        int y = a.*q;
        // since-cxx11-error@-1 {{reference to local variable 'a' declared in enclosing function 'cwg2083::volatile_lval'}}
        //   since-cxx11-note@#cwg2083-a2 {{'a' declared here}}
      }
    };
  }
#endif

  void discarded_lval() {
    struct A { int x; mutable int y; volatile int z; };
    A a; // #cwg2083-a-3
    int &r = a.x; // #cwg2083-r
    struct B {
      void f() {
        // FIXME: We emit more errors than we should be. They are explicitly
        // marked below.
        a.x;
        // expected-warning@-1 {{expression result unused}}
        // expected-error@-2 {{reference to local variable 'a' declared in enclosing function 'cwg2083::discarded_lval'}} FIXME
        //   expected-note@#cwg2083-a-3 {{'a' declared here}}
        a.*&A::x;
        // expected-warning@-1 {{expression result unused}}
        // expected-error@-2 {{reference to local variable 'a' declared in enclosing function 'cwg2083::discarded_lval'}} FIXME
        //   expected-note@#cwg2083-a-3 {{'a' declared here}}
        true ? a.x : a.y; // #cwg2083-ternary
        // expected-warning@-1 {{expression result unused}}
        // expected-error@#cwg2083-ternary {{reference to local variable 'a' declared in enclosing function 'cwg2083::discarded_lval'}} FIXME
        //   expected-note@#cwg2083-a-3 {{'a' declared here}}
        // expected-error@#cwg2083-ternary {{reference to local variable 'a' declared in enclosing function 'cwg2083::discarded_lval'}} FIXME
        //   expected-note@#cwg2083-a-3 {{'a' declared here}}
        (void)a.x;
        // expected-error@-1 {{reference to local variable 'a' declared in enclosing function 'cwg2083::discarded_lval'}} FIXME
        //   expected-note@#cwg2083-a-3 {{'a' declared here}}
        a.x, discarded_lval();
        // expected-warning@-1 {{left operand of comma operator has no effect}}
        // expected-error@-2 {{reference to local variable 'a' declared in enclosing function 'cwg2083::discarded_lval'}} FIXME
        //   expected-note@#cwg2083-a-3 {{'a' declared here}}

        // 'volatile' qualifier triggers an lvalue-to-rvalue conversion.
        a.z;
        // cxx98-warning@-1 {{expression result unused; assign into a variable to force a volatile load}}
        // expected-error@-2 {{reference to local variable 'a' declared in enclosing function 'cwg2083::discarded_lval'}}
        //   expected-note@#cwg2083-a-3 {{'a' declared here}}

        // References always get "loaded" to determine what they reference,
        // even if the result is discarded.
        r;
        // expected-warning@-1 {{expression result unused}}
        // expected-error@-2 {{reference to local variable 'r' declared in enclosing function 'cwg2083::discarded_lval'}}
        //   expected-note@#cwg2083-r {{'r' declared here}}
      }
    };
  }

  namespace dr_example_1 {
    extern int globx;
    int main() {
      const int &x = globx; // #cwg2083-x
      struct A {
        const int *foo() { return &x; }
        // cxx98-error@-1 {{reference to local variable 'x' declared in enclosing function 'cwg2083::dr_example_1::main'}}
        //   cxx98-note@#cwg2083-x {{'x' declared here}}
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
} // namespace cwg2083

namespace cwg2084 { // cwg2084: 3.1
struct S {
  S();
};
union U {
  S s{}; // cxx98-error {{function definition does not declare parameters}}
} u;
} // namespace cwg2084

namespace cwg2091 { // cwg2091: 10
template<int &> struct X;
template<int &N> void f(X<N>&);
int n;
void g(X<n> &x) { f(x); }

namespace GH42233 {
enum E { I };

class AA { };
E EV[1] = {I};

template<class ENUM, const ENUM* const VALUES>
struct S
{
  template< class E, const E* const V>
    friend AA& operator<<( AA& os, const S<E,V>& e );
};

int f()
{
  S< E, EV > x;

  AA a;
  a << x;
  return 0;
}
} // namespace GH42233
} // namespace cwg2091 

namespace cwg2094 { // cwg2094: 5
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
} // namespace cwg2094

// cwg2096: dup 2598
