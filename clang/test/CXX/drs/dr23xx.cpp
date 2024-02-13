// RUN: %clang_cc1 -std=c++98 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,since-cxx11,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx11,since-cxx14,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx11,since-cxx14,since-cxx17,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx11,since-cxx14,since-cxx17,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx11,since-cxx14,since-cxx17,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s

#if __cplusplus >= 201103L
namespace dr2303 { // dr2303: 12
template <typename... T>
struct A;
template <>
struct A<> {};
template <typename T, typename... Ts>
struct A<T, Ts...> : A<Ts...> {};
struct B : A<int, int> {};
struct C : A<int, int>, A<int> {};
/* since-cxx11-warning@-1 {{direct base 'A<int>' is inaccessible due to ambiguity:
    struct dr2303::C -> A<int, int> -> A<int>
    struct dr2303::C -> A<int>}} */
struct D : A<int>, A<int, int> {};
/* since-cxx11-warning@-1 {{direct base 'A<int>' is inaccessible due to ambiguity:
    struct dr2303::D -> A<int>
    struct dr2303::D -> A<int, int> -> A<int>}} */
struct E : A<int, int> {};
struct F : B, E {};

template <typename... T>
void f(const A<T...> &) {
  static_assert(sizeof...(T) == 2, "Should only match A<int,int>");
}
template <typename... T>
void f2(const A<T...> *);

void g() {
  f(B{}); // This is no longer ambiguous.
  B b;
  f2(&b);
  f(C{});
  f(D{});
  f(F{});
  /* since-cxx11-error@-1 {{ambiguous conversion from derived class 'const F' to base class 'const A<int, int>':
    struct dr2303::F -> B -> A<int, int>
    struct dr2303::F -> E -> A<int, int>}} */
}
} // namespace dr2303
#endif

// dr2331: na
// dr2335 is in dr2335.cxx

#if __cplusplus >= 201103L
namespace dr2338 { // dr2338: 12
namespace B {
enum E : bool { Zero, One };
static_assert((int)(E)2 == 1, "");
} // namespace B
namespace D {
enum class E : bool { Zero, One };
static_assert((int)(E)2 == 1, "");
} // namespace D
} // namespace dr2338
#endif

namespace dr2346 { // dr2346: 11
  void test() {
    const int i2 = 0;
    extern void h2b(int x = i2 + 0); // ok, not odr-use
  }
}

namespace dr2352 { // dr2352: 10
  int **p;
  const int *const *const &f1() { return p; }
  int *const *const &f2() { return p; }
  int **const &f3() { return p; }

  const int **const &f4() { return p; }
  // expected-error@-1 {{reference to type 'const int **const' could not bind to an lvalue of type 'int **'}}
  const int *const *&f5() { return p; }
  // expected-error@-1 {{binding reference of type 'const int *const *' to value of type 'int **' not permitted due to incompatible qualifiers}}

  // FIXME: We permit this as a speculative defect resolution, allowing
  // qualification conversions when forming a glvalue conditional expression.
  const int * const * const q = 0;
  __typeof(&(true ? p : q)) x = &(true ? p : q);

  // FIXME: Should we compute the composite pointer type here and produce an
  // lvalue of type 'const int *const * const'?
  const int * const * r;
  void *y = &(true ? p : r);
  // expected-error@-1 {{rvalue of type 'const int *const *'}}

  // FIXME: We order these as a speculative defect resolution.
  void f(const int * const * const &r);
#if __cplusplus >= 201103L
  constexpr
#endif
  int *const *const &f(int * const * const &r) { return r; }

  // No temporary is created here.
  int *const *const &check_f = f(p);
#if __cplusplus >= 201103L
  static_assert(&p == &check_f, "");
#endif
}

namespace dr2353 { // dr2353: 9
  struct X {
    static const int n = 0;
  };

  // CHECK: FunctionDecl {{.*}} use
  int use(X x) {
    // CHECK: MemberExpr {{.*}} .n
    // CHECK-NOT: non_odr_use
    // CHECK: DeclRefExpr {{.*}} 'x'
    // CHECK-NOT: non_odr_use
    return *&x.n;
  }
#pragma clang __debug dump use

  // CHECK: FunctionDecl {{.*}} not_use
  int not_use(X x) {
    // CHECK: MemberExpr {{.*}} .n {{.*}} non_odr_use_constant
    // CHECK: DeclRefExpr {{.*}} 'x'
    return x.n;
  }
#pragma clang __debug dump not_use

  // CHECK: FunctionDecl {{.*}} not_use_2
  int not_use_2(X *x) {
    // CHECK: MemberExpr {{.*}} ->n {{.*}} non_odr_use_constant
    // CHECK: DeclRefExpr {{.*}} 'x'
    return x->n;
  }
#pragma clang __debug dump not_use_2
}

namespace dr2354 { // dr2354: 15
#if __cplusplus >= 201103L
enum alignas(64) A {};
// since-cxx11-error@-1 {{'alignas' attribute cannot be applied to an enumeration}}
enum struct alignas(64) B {};
// since-cxx11-error@-1 {{'alignas' attribute cannot be applied to an enumeration}}
#endif
} // namespace dr2354

namespace dr2356 { // dr2356: 4
#if __cplusplus >= 201103L
struct A {
  A();
  A(A &&);                        // #1
  template<typename T> A(T &&);   // #2
};
struct B : A {
  using A::A;
  B(const B &);                   // #3
  B(B &&) = default;              // #4, implicitly deleted
  // since-cxx11-warning@-1 {{explicitly defaulted move constructor is implicitly deleted}}
  //   since-cxx11-note@#dr2356-X {{move constructor of 'B' is implicitly deleted because field 'x' has a deleted move constructor}}
  //   since-cxx11-note@#dr2356-X {{'X' has been explicitly marked deleted here}}
  //   since-cxx11-note@-4 {{replace 'default' with 'delete'}}

  struct X { X(X &&) = delete; } x; // #dr2356-X
};
extern B b1;
B b2 = static_cast<B&&>(b1);      // calls #3: #1, #2, and #4 are not viable
struct C { operator B&&(); };
B b3 = C();                       // calls #3
#endif
}

#if __cplusplus >= 201402L
namespace dr2358 { // dr2358: 16
  void f2() {
    int i = 1;
    void g1(int = [xxx=1] { return xxx; }());  // OK
    void g2(int = [xxx=i] { return xxx; }());
    // since-cxx14-error@-1 {{default argument references local variable 'i' of enclosing function}}
  }
}
#endif

namespace dr2370 { // dr2370: no
namespace N {
typedef int type;
void g(type);
void h(type);
} // namespace N
class C {
  typedef N::type N_type;
  // FIXME: `type` should be searched for in N
  // friend void N::g(type);
  friend void N::h(N_type);
};
} // namespace dr2370

#if __cplusplus >= 201702L
// Otherwise, if the qualified-id std::tuple_size<E> names a complete class
// type **with a member value**, the expression std::tuple_size<E>::value shall
// be a well-formed integral constant expression
namespace dr2386 { // dr2386: 9
struct Bad1 { int a, b; };
struct Bad2 { int a, b; };
} // namespace dr2386
namespace std {
template <typename T> struct tuple_size;
template <> struct tuple_size<dr2386::Bad1> {};
template <> struct tuple_size<dr2386::Bad2> {
  static const int value = 42;
};
} // namespace std
namespace dr2386 {
void no_value() { auto [x, y] = Bad1(); }
void wrong_value() { auto [x, y] = Bad2(); }
// since-cxx17-error@-1 {{type 'Bad2' decomposes into 42 elements, but only 2 names were provided}}
} // namespace dr2386
#endif

// dr2385: na

namespace dr2387 { // dr2387: 9
#if __cplusplus >= 201402L
  template<int> int a = 0;
  extern template int a<0>; // ok

  template<int> static int b = 0;
  extern template int b<0>;
  // since-cxx14-error@-1 {{explicit instantiation declaration of 'b<0>' with internal linkage}}

  template<int> const int c = 0;
  extern template const int c<0>; // ok, has external linkage despite 'const'

  template<typename T> T d = 0;
  extern template int d<int>;
  extern template const int d<const int>;
#endif
}

// dr2390 is in dr2390.cpp

namespace dr2394 { // dr2394: 15

struct A {};
const A a;

// Now allowed to default-init B.
struct B { const A a; };
B b;

}

namespace dr2396 { // dr2396: no
  struct A {
    struct B;
    operator B B::*();
  };
  struct B;

  // FIXME: per P1787 "Calling a conversion function" example, all of the
  // examples below are well-formed, with B resolving to A::B, but currently
  // it's been resolved to dr2396::B.

  // void f(A a) { a.operator B B::*(); }
  // void g(A a) { a.operator decltype(B()) B::*(); }
  // void g2(A a) { a.operator B decltype(B())::*(); }
}

#if __cplusplus >= 201103L
namespace dr2397 { // dr2397: 17
  void foo() {
    int a[5];

    auto (&b)[5] = a;
    auto (*c)[5] = &a;
  }
} // namespace dr2397

// CWG2363 was closed as NAD, but its resolution does affirm that
// a friend declaration cannot have an opaque-enumm-specifier.
namespace dr2363 { // dr2363: yes

enum class E0;
enum E1 : int;

struct A {
  friend enum class E0;
  // since-cxx11-error@-1 {{reference to enumeration must use 'enum' not 'enum class'}}
  // expected-error@-2 {{elaborated enum specifier cannot be declared as a friend}}
  // expected-note@-3 {{remove 'enum class' to befriend an enum}}

  friend enum E0;
  // expected-error@-1 {{elaborated enum specifier cannot be declared as a friend}}
  // expected-note@-2 {{remove 'enum' to befriend an enum}}

  friend enum class E1;
  // since-cxx11-error@-1 {{reference to enumeration must use 'enum' not 'enum class'}}
  // expected-error@-2 {{elaborated enum specifier cannot be declared as a friend}}
  // expected-note@-3 {{remove 'enum class' to befriend an enum}}

  friend enum E1;
  // expected-error@-1 {{elaborated enum specifier cannot be declared as a friend}}
  // expected-note@-2 {{remove 'enum' to befriend an enum}}

  friend enum class E2;
  // since-cxx11-error@-1 {{reference to enumeration must use 'enum' not 'enum class'}}
  // expected-error@-2 {{elaborated enum specifier cannot be declared as a friend}}
  // expected-note@-3 {{remove 'enum class' to befriend an enum}}
};
} // namespace dr2363

#endif
