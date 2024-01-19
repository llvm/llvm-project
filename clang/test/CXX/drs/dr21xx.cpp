// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected,cxx98-14,cxx98 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,cxx98-14,since-cxx11 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,cxx98-14,since-cxx11 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -Wno-deprecated-builtins -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

namespace std {
  __extension__ typedef __SIZE_TYPE__ size_t;

  template<typename E> struct initializer_list {
    const E *p; size_t n;
    initializer_list(const E *p, size_t n);
    initializer_list();
  };
}

namespace dr2100 { // dr2100: 12
  template<const int *P, bool = true> struct X {};
  template<typename T> struct A {
    static const int n = 1;
    int f() {
      return X<&n>::n; // ok, value-dependent
    }
    int g() {
      static const int n = 2; // #dr2100-n
      return X<&n>::n; // ok, value-dependent
      // cxx98-14-error@-1 {{non-type template argument refers to object 'n' that does not have linkage}}
      //   cxx98-14-note@#dr2100-n {{non-type template argument refers to object here}}
    }
  };
  template<const int *P> struct X<P> {
#if __cplusplus == 199711L
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
      return Y<n>::declared_later;
      // expected-error@-1 {{no member named 'declared_later' in 'dr2100::Y<1>'}}
    }
    int g() {
      static const int n = 2;
      return Y<n>::declared_later;
      // expected-error@-1 {{no member named 'declared_later' in 'dr2100::Y<2>'}}
    }
  };
  template<int N> struct Y<N> {
    static const int declared_later = 0;
  };
}

namespace dr2103 { // dr2103: yes
  void f() {
    int a;
    int &r = a; // #dr2103-r
    struct Inner {
      void f() {
        int &s = r;
        // expected-error@-1 {{reference to local variable 'r' declared in enclosing function 'dr2103::f'}}
        //   expected-note@#dr2103-r {{'r' declared here}}
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

  A &&d = {1};                   // non-const temporary #dr21260-d
  const A &e = (A &)(A &&) A{1}; // non-const temporary #dr21260-e
  A &&f = (A &&)(A &&) A{1};     // non-const temporary #dr21260-f

  constexpr const A &g = {1};    // const temporary
  constexpr A &&h = {1};         // non-const temporary #dr21260-h

  struct B { const A &a; };
  B i = {{1}};           // extending decl not usable in constant expr #dr21260-i
  const B j = {{1}};     // extending decl not usable in constant expr #dr21260-j
  constexpr B k = {{1}}; // extending decl usable in constant expr

  static_assert(a.n == 1, "");
  static_assert(b.n == 1, "");
  static_assert(c.n == 1, "");
  static_assert(d.n == 1, "");
  // since-cxx11-error@-1 {{static assertion expression is not an integral constant expression}}
  //   since-cxx11-note@-2 {{read of temporary is not allowed in a constant expression outside the expression that created the temporary}}
  //   since-cxx11-note@#dr21260-d {{temporary created here}}
  static_assert(e.n == 1, "");
  // since-cxx11-error@-1 {{static assertion expression is not an integral constant expression}}
  //   since-cxx11-note@-2 {{read of temporary is not allowed in a constant expression outside the expression that created the temporary}}
  //   since-cxx11-note@#dr21260-e {{temporary created here}}
  static_assert(f.n == 1, "");
  // since-cxx11-error@-1 {{static assertion expression is not an integral constant expression}}
  //   since-cxx11-note@-2 {{read of temporary is not allowed in a constant expression outside the expression that created the temporary}}
  //   since-cxx11-note@#dr21260-f {{temporary created here}}
  static_assert(g.n == 1, "");
  static_assert(h.n == 1, "");
  // since-cxx11-error@-1 {{static assertion expression is not an integral constant expression}}
  //   since-cxx11-note@-2 {{read of temporary is not allowed in a constant expression outside the expression that created the temporary}}
  //   since-cxx11-note@#dr21260-h {{temporary created here}}
  static_assert(i.a.n == 1, "");
  // since-cxx11-error@-1 {{static assertion expression is not an integral constant expression}}
  //   since-cxx11-note@-2 {{read of non-constexpr variable 'i' is not allowed in a constant expression}}
  //   since-cxx11-note@#dr21260-i {{declared here}}
  static_assert(j.a.n == 1, "");
  // since-cxx11-error@-1 {{static assertion expression is not an integral constant expression}}
  //   since-cxx11-note@-2 {{read of temporary is not allowed in a constant expression outside the expression that created the temporary}}
  //   since-cxx11-note@#dr21260-j {{temporary created here}}
  static_assert(k.a.n == 1, "");
#endif
}

namespace dr2137 { // dr2137: 18
#if __cplusplus >= 201103L
  struct Q {
    Q();
    Q(Q&&);
    Q(std::initializer_list<Q>) = delete; // #dr2137-Qcons
  };

  Q x = Q { Q() };
  // since-cxx11-error@-1 {{call to deleted constructor of 'Q'}}
  //   since-cxx11-note@#dr2137-Qcons {{'Q' has been explicitly marked deleted here}}

  int f(Q); // #dr2137-f
  int y = f({ Q() });
  // since-cxx11-error@-1 {{call to deleted constructor of 'Q'}}
  //   since-cxx11-note@#dr2137-Qcons {{'Q' has been explicitly marked deleted here}}
  //   since-cxx11-note@#dr2137-f {{passing argument to parameter here}}

  struct U {
    U();
    U(const U&);
  };

  struct Derived : U {
    Derived();
    Derived(const Derived&);
  } d;

  int g(Derived);
  int g(U(&&)[1]) = delete;

  int z = g({ d });
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

namespace dr2141 { // dr2141: 17
struct A{};

template <typename T>
struct B{};

void foo() {
  struct A *b = (1 == 1) ? new struct A : new struct A;
  struct S *a = (1 == 1) ? new struct S : new struct S;
  // expected-error@-1 {{allocation of incomplete type 'struct S'}}
  //   expected-note@-2 {{forward declaration of 'S'}}
  // expected-error@-3 {{allocation of incomplete type 'struct S'}}
  //   expected-note@-4 {{forward declaration of 'S'}}

#if __cplusplus >= 201103L
  A *aa = new struct A{};
  B<int> *bb = new struct B<int>{};
  (void)new struct C{};
  // since-cxx11-error@-1 {{allocation of incomplete type 'struct C'}}
  //   since-cxx11-note@-2 {{forward declaration of 'C'}}

  struct A *c = (1 == 1) ? new struct A {} : new struct A {};

  alignof(struct D{});
  // since-cxx11-error@-1 {{'D' cannot be defined in a type specifier}}
#endif

  sizeof(struct E{});
  // expected-error@-1 {{'E' cannot be defined in a type specifier}}

}
}

namespace dr2157 { // dr2157: 11
#if __cplusplus >= 201103L
  enum E : int;
  struct X {
    enum dr2157::E : int();
    // since-cxx11-error@-1 {{ISO C++ only allows ':' in member enumeration declaration to introduce a fixed underlying type, not an anonymous bit-field}}
  };
#endif
}

// dr2165: na

namespace dr2170 { // dr2170: 9
#if __cplusplus >= 201103L
  void f() {
    constexpr int arr[3] = {1, 2, 3}; // #dr2170-arr
    struct S {
      int get(int n) { return arr[n]; }
      const int &get_ref(int n) { return arr[n]; }
      // since-cxx11-warning@-1 {{reference to stack memory associated with local variable 'arr' returned}} FIXME
      // since-cxx11-error@-2 {{reference to local variable 'arr' declared in enclosing function 'dr2170::f'}}
      //   since-cxx11-note@#dr2170-arr {{'arr' declared here}}
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
    A &operator=(const A &); // #dr2180-A-copy
    A &operator=(A &&); // #dr2180-A-move
    // cxx98-error@-1 {{rvalue references are a C++11 extension}}
  };

  struct B : virtual A { // #dr2180-B
    B &operator=(const B &);
    B &operator=(B &&);
    // cxx98-error@-1 {{rvalue references are a C++11 extension}}
    virtual void foo() = 0;
  };
  B &B::operator=(const B&) = default; // #dr2180-B-copy
  // cxx98-error@-1 {{defaulted function definitions are a C++11 extension}}
  // cxx98-error@-2 {{'operator=' is a private member of 'dr2180::A'}}
  //   cxx98-note@-3 {{in defaulted copy assignment operator for 'dr2180::B' first required here}}
  //   cxx98-note@#dr2180-A-copy {{implicitly declared private here}}
  // since-cxx11-error@#dr2180-B-copy {{defaulting this copy assignment operator would delete it after its first declaration}}
  //   since-cxx11-note@#dr2180-B {{copy assignment operator of 'B' is implicitly deleted because base class 'A' has an inaccessible copy assignment operator}}
  B &B::operator=(B&&) = default; // #dr2180-B-move
  // cxx98-error@-1 {{rvalue references are a C++11 extension}}
  // cxx98-error@-2 {{defaulted function definitions are a C++11 extension}}
  // cxx98-error@-3 {{'operator=' is a private member of 'dr2180::A'}}
  //   cxx98-note@-4 {{in defaulted move assignment operator for 'dr2180::B' first required here}}
  //   cxx98-note@#dr2180-A-move {{implicitly declared private here}}
  // since-cxx11-error@#dr2180-B-move {{defaulting this move assignment operator would delete it after its first declaration}}
  //   since-cxx11-note@#dr2180-B {{move assignment operator of 'B' is implicitly deleted because base class 'A' has an inaccessible move assignment operator}}
}

namespace dr2199 { // dr2199: 3.8
                   // NB: reusing part of dr407 test
namespace A {
  struct S {};
}
namespace B {
  typedef int S;
}
namespace E {
  typedef A::S S;
  using A::S;
  struct S s;
}
namespace F {
  typedef A::S S;
}
namespace G {
  using namespace A;
  using namespace F;
  struct S s;
}
namespace H {
  using namespace F;
  using namespace A;
  struct S s;
}
}
