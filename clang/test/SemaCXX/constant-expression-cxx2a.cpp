// RUN: %clang_cc1 -std=c++2a -verify %s -fcxx-exceptions -triple=x86_64-linux-gnu

#include "Inputs/std-compare.h"

namespace ThreeWayComparison {
  struct A {
    int n;
    constexpr friend int operator<=>(const A &a, const A &b) {
      return a.n < b.n ? -1 : a.n > b.n ? 1 : 0;
    }
  };
  static_assert(A{1} <=> A{2} < 0);
  static_assert(A{2} <=> A{1} > 0);
  static_assert(A{2} <=> A{2} == 0);

  static_assert(1 <=> 2 < 0);
  static_assert(2 <=> 1 > 0);
  static_assert(1 <=> 1 == 0);
  constexpr int k = (1 <=> 1, 0);
  // expected-warning@-1 {{three-way comparison result unused}}

  static_assert(std::strong_ordering::equal == 0);

  constexpr void f() {
    void(1 <=> 1);
  }

  struct MemPtr {
    void foo() {}
    void bar() {}
    int data;
    int data2;
    long data3;
  };

  struct MemPtr2 {
    void foo() {}
    void bar() {}
    int data;
    int data2;
    long data3;
  };
  using MemPtrT = void (MemPtr::*)();

  using FnPtrT = void (*)();

  void FnPtr1() {}
  void FnPtr2() {}

#define CHECK(...) ((__VA_ARGS__) ? void() : throw "error")
#define CHECK_TYPE(...) static_assert(__is_same(__VA_ARGS__));

constexpr bool test_constexpr_success = [] {
  {
    auto &EQ = std::strong_ordering::equal;
    auto &LESS = std::strong_ordering::less;
    auto &GREATER = std::strong_ordering::greater;
    using SO = std::strong_ordering;
    auto eq = (42 <=> 42);
    CHECK_TYPE(decltype(eq), SO);
    CHECK(eq.test_eq(EQ));

    auto less = (-1 <=> 0);
    CHECK_TYPE(decltype(less), SO);
    CHECK(less.test_eq(LESS));

    auto greater = (42l <=> 1u);
    CHECK_TYPE(decltype(greater), SO);
    CHECK(greater.test_eq(GREATER));
  }
  {
    using PO = std::partial_ordering;
    auto EQUIV = PO::equivalent;
    auto LESS = PO::less;
    auto GREATER = PO::greater;

    auto eq = (42.0 <=> 42.0);
    CHECK_TYPE(decltype(eq), PO);
    CHECK(eq.test_eq(EQUIV));

    auto less = (39.0 <=> 42.0);
    CHECK_TYPE(decltype(less), PO);
    CHECK(less.test_eq(LESS));

    auto greater = (-10.123 <=> -101.1);
    CHECK_TYPE(decltype(greater), PO);
    CHECK(greater.test_eq(GREATER));
  }
  {
    using SE = std::strong_equality;
    auto EQ = SE::equal;
    auto NEQ = SE::nonequal;

    MemPtrT P1 = &MemPtr::foo;
    MemPtrT P12 = &MemPtr::foo;
    MemPtrT P2 = &MemPtr::bar;
    MemPtrT P3 = nullptr;

    auto eq = (P1 <=> P12);
    CHECK_TYPE(decltype(eq), SE);
    CHECK(eq.test_eq(EQ));

    auto neq = (P1 <=> P2);
    CHECK_TYPE(decltype(eq), SE);
    CHECK(neq.test_eq(NEQ));

    auto eq2 = (P3 <=> nullptr);
    CHECK_TYPE(decltype(eq2), SE);
    CHECK(eq2.test_eq(EQ));
  }
  {
    using SE = std::strong_equality;
    auto EQ = SE::equal;
    auto NEQ = SE::nonequal;

    FnPtrT F1 = &FnPtr1;
    FnPtrT F12 = &FnPtr1;
    FnPtrT F2 = &FnPtr2;
    FnPtrT F3 = nullptr;

    auto eq = (F1 <=> F12);
    CHECK_TYPE(decltype(eq), SE);
    CHECK(eq.test_eq(EQ));

    auto neq = (F1 <=> F2);
    CHECK_TYPE(decltype(neq), SE);
    CHECK(neq.test_eq(NEQ));
  }
  { // mixed nullptr tests
    using SO = std::strong_ordering;
    using SE = std::strong_equality;

    int x = 42;
    int *xp = &x;

    MemPtrT mf = nullptr;
    MemPtrT mf2 = &MemPtr::foo;
    auto r3 = (mf <=> nullptr);
    CHECK_TYPE(decltype(r3), std::strong_equality);
    CHECK(r3.test_eq(SE::equal));
  }

  return true;
}();

template <auto LHS, auto RHS, bool ExpectTrue = false>
constexpr bool test_constexpr() {
  using nullptr_t = decltype(nullptr);
  using LHSTy = decltype(LHS);
  using RHSTy = decltype(RHS);
  // expected-note@+1 {{subexpression not valid in a constant expression}}
  auto Res = (LHS <=> RHS);
  if constexpr (__is_same(LHSTy, nullptr_t) || __is_same(RHSTy, nullptr_t)) {
    CHECK_TYPE(decltype(Res), std::strong_equality);
  }
  if (ExpectTrue)
    return Res == 0;
  return Res != 0;
}
int dummy = 42;
int dummy2 = 101;

constexpr bool tc1 = test_constexpr<nullptr, &dummy>();
constexpr bool tc2 = test_constexpr<&dummy, nullptr>();

// OK, equality comparison only
constexpr bool tc3 = test_constexpr<&MemPtr::foo, nullptr>();
constexpr bool tc4 = test_constexpr<nullptr, &MemPtr::foo>();
constexpr bool tc5 = test_constexpr<&MemPtr::foo, &MemPtr::bar>();

constexpr bool tc6 = test_constexpr<&MemPtr::data, nullptr>();
constexpr bool tc7 = test_constexpr<nullptr, &MemPtr::data>();
constexpr bool tc8 = test_constexpr<&MemPtr::data, &MemPtr::data2>();

// expected-error@+1 {{must be initialized by a constant expression}}
constexpr bool tc9 = test_constexpr<&dummy, &dummy2>(); // expected-note {{in call}}

template <class T, class R, class I>
constexpr T makeComplex(R r, I i) {
  T res{r, i};
  return res;
};

template <class T, class ResultT>
constexpr bool complex_test(T x, T y, ResultT Expect) {
  auto res = x <=> y;
  CHECK_TYPE(decltype(res), ResultT);
  return res.test_eq(Expect);
}
static_assert(complex_test(makeComplex<_Complex double>(0.0, 0.0),
                           makeComplex<_Complex double>(0.0, 0.0),
                           std::weak_equality::equivalent));
static_assert(complex_test(makeComplex<_Complex double>(0.0, 0.0),
                           makeComplex<_Complex double>(1.0, 0.0),
                           std::weak_equality::nonequivalent));
static_assert(complex_test(makeComplex<_Complex double>(0.0, 0.0),
                           makeComplex<_Complex double>(0.0, 1.0),
                           std::weak_equality::nonequivalent));
static_assert(complex_test(makeComplex<_Complex int>(0, 0),
                           makeComplex<_Complex int>(0, 0),
                           std::strong_equality::equal));
static_assert(complex_test(makeComplex<_Complex int>(0, 0),
                           makeComplex<_Complex int>(1, 0),
                           std::strong_equality::nonequal));
// TODO: defaulted operator <=>
} // namespace ThreeWayComparison
