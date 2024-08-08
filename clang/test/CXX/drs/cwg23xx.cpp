// RUN: %clang_cc1 -std=c++98 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx11-14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,cxx11-14,since-cxx11,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx11,since-cxx14,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx11,since-cxx14,since-cxx17,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx11,since-cxx14,since-cxx17,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx11,since-cxx14,since-cxx17,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s

namespace std {
  __extension__ typedef __SIZE_TYPE__ size_t;

  template<typename E> struct initializer_list {
    const E *p; size_t n;
    initializer_list(const E *p, size_t n);
    initializer_list();
  };
}

#if __cplusplus >= 201103L
namespace cwg2303 { // cwg2303: 12
template <typename... T>
struct A;
template <>
struct A<> {};
template <typename T, typename... Ts>
struct A<T, Ts...> : A<Ts...> {};
struct B : A<int, int> {};
struct C : A<int, int>, A<int> {};
/* since-cxx11-warning@-1 {{direct base 'A<int>' is inaccessible due to ambiguity:
    struct cwg2303::C -> A<int, int> -> A<int>
    struct cwg2303::C -> A<int>}} */
struct D : A<int>, A<int, int> {};
/* since-cxx11-warning@-1 {{direct base 'A<int>' is inaccessible due to ambiguity:
    struct cwg2303::D -> A<int>
    struct cwg2303::D -> A<int, int> -> A<int>}} */
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
    struct cwg2303::F -> B -> A<int, int>
    struct cwg2303::F -> E -> A<int, int>}} */
}
} // namespace cwg2303
#endif

namespace cwg2304 { // cwg2304: 2.8
template<typename T> void foo(T, int);
template<typename T> void foo(T&, ...);
struct Q; // #cwg2304-Q
void fn1(Q &data_vectors) {
  foo(data_vectors, 0);
  // expected-error@-1 {{argument type 'cwg2304::Q' is incomplete}}
  //   expected-note@#cwg2304-Q {{forward declaration of 'cwg2304::Q'}}
}
} // namespace cwg2304

namespace cwg2310 { // cwg2310: partial
#if __cplusplus >= 201103L
template<typename A, typename B>
struct check_derived_from {
  static A a;
  // FIXME: all 3 examples should be rejected in all language modes.
  // FIXME: we should test this in 98 mode.
  // FIXME: we accept this when MSVC triple is used
  static constexpr B *p = &a;
#if !defined(_WIN32) || defined(__MINGW32__)
  // cxx11-14-error@-2 {{cannot initialize a variable of type 'cwg2310::X *const' with an rvalue of type 'cwg2310::Z *'}}
  //   cxx11-14-note@#cwg2310-X {{in instantiation of template class 'cwg2310::check_derived_from<cwg2310::Z, cwg2310::X>' requested here}}
  // cxx11-14-error@-4 {{cannot initialize a variable of type 'cwg2310::Y *const' with an rvalue of type 'cwg2310::Z *'}}
  //   cxx11-14-note@#cwg2310-Y {{in instantiation of template class 'cwg2310::check_derived_from<cwg2310::Z, cwg2310::Y>' requested here}}
#endif
};

struct W {};
struct X {};
struct Y {};
struct Z : W,
  X, check_derived_from<Z, X>, // #cwg2310-X
  check_derived_from<Z, Y>, Y  // #cwg2310-Y
{  
  // FIXME: It was properly rejected before, but we're crashing since Clang 11 in C++11 and C++14 modes.
  //        See https://github.com/llvm/llvm-project/issues/59920
#if __cplusplus >= 201703L
  check_derived_from<Z, W> cdf;
#endif
};
#endif
} // namespace cwg2310

// cwg2331: na
// cwg2335 is in cwg2335.cxx

namespace cwg2311 {  // cwg2311 is open with no proposed resolution
#if __cplusplus >= 201707L
template<typename T>
void test() {
  // Ensure none of these try to call a move constructor.
  T a = T{T(0)};
  T b{T(0)};
  auto c{T(0)};
  T d = {T(0)};
  auto e = {T(0)};
#if __cplusplus >= 202302L
  auto f = auto{T(0)};
#endif
  void(*fn)(T);
  fn({T(0)});
}

struct NonMovable {
  NonMovable(int);
  NonMovable(NonMovable&&) = delete;
};
struct NonMovableNonApplicableIList {
  NonMovableNonApplicableIList(int);
  NonMovableNonApplicableIList(NonMovableNonApplicableIList&&) = delete;
  NonMovableNonApplicableIList(std::initializer_list<int>);
};
struct ExplicitMovable {
  ExplicitMovable(int);
  explicit ExplicitMovable(ExplicitMovable&&);
};
struct ExplicitNonMovable {
  ExplicitNonMovable(int);
  explicit ExplicitNonMovable(ExplicitNonMovable&&) = delete;
};
struct ExplicitNonMovableNonApplicableIList {
  ExplicitNonMovableNonApplicableIList(int);
  explicit ExplicitNonMovableNonApplicableIList(ExplicitNonMovableNonApplicableIList&&) = delete;
  ExplicitNonMovableNonApplicableIList(std::initializer_list<int>);
};
struct CopyOnly {
  CopyOnly(int);
  CopyOnly(const CopyOnly&);
  CopyOnly(CopyOnly&&) = delete;
};
struct ExplicitCopyOnly {
  ExplicitCopyOnly(int);
  explicit ExplicitCopyOnly(const ExplicitCopyOnly&);
  explicit ExplicitCopyOnly(ExplicitCopyOnly&&) = delete;
};

template void test<NonMovable>();
template void test<NonMovableNonApplicableIList>();
template void test<ExplicitMovable>();
template void test<ExplicitNonMovable>();
template void test<ExplicitNonMovableNonApplicableIList>();
template void test<CopyOnly>();
template void test<ExplicitCopyOnly>();

struct any {
    template<typename T>
    any(T&&);
};

template<typename T>
struct X {
    X();
    X(T) = delete; // #cwg2311-X
};

X<std::initializer_list<any>> x{ X<std::initializer_list<any>>() };
// since-cxx17-error@-1 {{call to deleted constructor of 'X<std::initializer_list<any>>'}}
//   since-cxx17-note@#cwg2311-X {{'X' has been explicitly marked deleted here}}

// Per the currently implemented resolution, this does not apply to std::initializer_list.
// An initializer list initialized from `{ e }` always has exactly one element constructed
// from `e`, where previously that could have been a copy of an init list or `e.operator std::initializer_list()`
struct InitListCtor {
  InitListCtor(int);
  InitListCtor(InitListCtor&&) = delete;
  InitListCtor(std::initializer_list<InitListCtor>) = delete; // #cwg2311-InitListCtor
};

std::initializer_list<InitListCtor> i;
auto j = std::initializer_list<InitListCtor>{ i };
// since-cxx17-error@-1 {{conversion function from 'std::initializer_list<InitListCtor>' to 'const cwg2311::InitListCtor' invokes a deleted function}}
//   since-cxx17-note@#cwg2311-InitListCtor {{'InitListCtor' has been explicitly marked deleted here}}
#endif
}

#if __cplusplus >= 201103L
namespace cwg2338 { // cwg2338: 12
namespace B {
enum E : bool { Zero, One };
static_assert((int)(E)2 == 1, "");
} // namespace B
namespace D {
enum class E : bool { Zero, One };
static_assert((int)(E)2 == 1, "");
} // namespace D
} // namespace cwg2338
#endif

namespace cwg2346 { // cwg2346: 11
  void test() {
    const int i2 = 0;
    extern void h2b(int x = i2 + 0); // ok, not odr-use
  }
}

namespace cwg2352 { // cwg2352: 10
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

namespace cwg2353 { // cwg2353: 9
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

namespace cwg2354 { // cwg2354: 15
#if __cplusplus >= 201103L
enum alignas(64) A {};
// since-cxx11-error@-1 {{'alignas' attribute cannot be applied to an enumeration}}
enum struct alignas(64) B {};
// since-cxx11-error@-1 {{'alignas' attribute cannot be applied to an enumeration}}
#endif
} // namespace cwg2354

namespace cwg2356 { // cwg2356: 4
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
  //   since-cxx11-note@#cwg2356-X {{move constructor of 'B' is implicitly deleted because field 'x' has a deleted move constructor}}
  //   since-cxx11-note@#cwg2356-X {{'X' has been explicitly marked deleted here}}
  //   since-cxx11-note@-4 {{replace 'default' with 'delete'}}

  struct X { X(X &&) = delete; } x; // #cwg2356-X
};
extern B b1;
B b2 = static_cast<B&&>(b1);      // calls #3: #1, #2, and #4 are not viable
struct C { operator B&&(); };
B b3 = C();                       // calls #3
#endif
}

#if __cplusplus >= 201402L
namespace cwg2358 { // cwg2358: 16
  void f2() {
    int i = 1;
    void g1(int = [xxx=1] { return xxx; }());  // OK
    void g2(int = [xxx=i] { return xxx; }());
    // since-cxx14-error@-1 {{default argument references local variable 'i' of enclosing function}}
  }
}
#endif

// CWG2363 was closed as NAD, but its resolution does affirm that
// a friend declaration cannot have an opaque-enumm-specifier.
namespace cwg2363 { // cwg2363: yes
#if __cplusplus >= 201103L
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
#endif
} // namespace cwg2363

namespace cwg2370 { // cwg2370: no
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
} // namespace cwg2370

#if __cplusplus >= 201702L
// Otherwise, if the qualified-id std::tuple_size<E> names a complete class
// type **with a member value**, the expression std::tuple_size<E>::value shall
// be a well-formed integral constant expression
namespace cwg2386 { // cwg2386: 9
struct Bad1 { int a, b; };
struct Bad2 { int a, b; };
} // namespace cwg2386
namespace std {
template <typename T> struct tuple_size;
template <> struct tuple_size<cwg2386::Bad1> {};
template <> struct tuple_size<cwg2386::Bad2> {
  static const int value = 42;
};
} // namespace std
namespace cwg2386 {
void no_value() { auto [x, y] = Bad1(); }
void wrong_value() { auto [x, y] = Bad2(); }
// since-cxx17-error@-1 {{type 'Bad2' decomposes into 42 elements, but only 2 names were provided}}
} // namespace cwg2386
#endif

// cwg2385: na

namespace cwg2387 { // cwg2387: 9
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

namespace cwg2390 { // cwg2390: 14
// Test that macro expansion of the builtin argument works.
#define C clang
#define F fallthrough
#define CF clang::fallthrough

#if !__has_cpp_attribute(F)
#error "doesn't have fallthrough"
#endif

#if !__has_cpp_attribute(C::F)
#error "doesn't have clang::fallthrough 1"
#endif

#if !__has_cpp_attribute(clang::F)
#error "doesn't have clang::fallthrough 2"
#endif

#if !__has_cpp_attribute(C::fallthrough)
#error "doesn't have clang::fallthrough 3"
#endif

#if !__has_cpp_attribute(CF)
#error "doesn't have clang::fallthrough 4"
#endif

#define FUNCLIKE1(x) clang::x
#if !__has_cpp_attribute(FUNCLIKE1(fallthrough))
#error "doesn't have clang::fallthrough through func-like macro 1"
#endif

#define FUNCLIKE2(x) _Clang::x
#if !__has_cpp_attribute(FUNCLIKE2(fallthrough))
#error "doesn't have clang::fallthrough through func-like macro 2"
#endif
} // namespace cwg2390

namespace cwg2394 { // cwg2394: 15

struct A {};
const A a;

// Now allowed to default-init B.
struct B { const A a; };
B b;

}

namespace cwg2396 { // cwg2396: no
  struct A {
    struct B;
    operator B B::*();
  };
  struct B;

  // FIXME: per P1787 "Calling a conversion function" example, all of the
  // examples below are well-formed, with B resolving to A::B, but currently
  // it's been resolved to cwg2396::B.

  // void f(A a) { a.operator B B::*(); }
  // void g(A a) { a.operator decltype(B()) B::*(); }
  // void g2(A a) { a.operator B decltype(B())::*(); }
}

#if __cplusplus >= 201103L
namespace cwg2397 { // cwg2397: 17
  void foo() {
    int a[5];

    auto (&b)[5] = a;
    auto (*c)[5] = &a;
  }
} // namespace cwg2397

#endif
