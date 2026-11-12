// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// Regression tests for https://github.com/llvm/llvm-project/issues/175934
// (and the same root cause as https://github.com/llvm/llvm-project/issues/110041).
//
// [temp.dep.constexpr]p5 makes `&x` value-dependent when `x` is a templated
// entity with static storage duration, even if `x` itself is not dependent.
// computeDependence(UnaryOperator*) used to set only the Value bit, leaving the
// expression not instantiation-dependent. decltype(&x) was then flagged as a
// dependent type but was never re-substituted on instantiation, so the
// uninstantiated type leaked into the specialization and asserted with
// "should not see dependent types here" in ASTContext::getTypeInfoImpl.

template <class T, class U> struct is_same { static const bool value = false; };
template <class T> struct is_same<T, T> { static const bool value = true; };

namespace field_of_class_template {
// The original reproducer.
template <class T> struct Y {
  static const int e = 1;
  decltype(&e) g;
};

Y<int> y; // Crashed here: constructor call checks compute the record layout.
static_assert(sizeof(Y<int>) == sizeof(const int *), "");
static_assert(alignof(Y<int>) == alignof(const int *), "");
// The field type must be re-evaluated for each specialization.
static_assert(is_same<decltype(Y<int>().g), const int *>::value, "");
static_assert(is_same<decltype(Y<char>().g), const int *>::value, "");
static_assert(sizeof(Y<char>) == sizeof(Y<int>), "");
} // namespace field_of_class_template

namespace constexpr_static_member {
template <class T> struct Y {
  static constexpr int e = 1;
  decltype(&e) g;
};
static_assert(sizeof(Y<int>) == sizeof(const int *), "");
static_assert(is_same<decltype(Y<int>().g), const int *>::value, "");
} // namespace constexpr_static_member

namespace array_static_member {
// &arr has type `const int (*)[2]`.
template <class T> struct Y {
  static constexpr int arr[2] = {1, 2};
  decltype(&arr) g;
};
static_assert(sizeof(Y<int>) == sizeof(void *), "");
static_assert(is_same<decltype(Y<int>().g), const int (*)[2]>::value, "");
} // namespace array_static_member

namespace via_alias_and_multiple_members {
template <class T> struct Y {
  static const int e = 1;
  using P = decltype(&e);
  typedef decltype(&e) Q;
  P a;
  Q b;
  decltype(&e) c;
};
static_assert(sizeof(Y<int>) == 3 * sizeof(const int *), "");
static_assert(is_same<Y<int>::P, const int *>::value, "");
static_assert(is_same<Y<int>::Q, const int *>::value, "");
} // namespace via_alias_and_multiple_members

namespace nested_layout {
// The bad field type must not poison records that contain the specialization.
template <class T> struct Y {
  static const int e = 1;
  decltype(&e) g;
};
struct Outer {
  char c;
  Y<int> y;
};
struct Derived : Y<long> {
  int i;
};
static_assert(sizeof(Outer) >= sizeof(char) + sizeof(const int *), "");
static_assert(sizeof(Derived) >= sizeof(const int *) + sizeof(int), "");
} // namespace nested_layout

namespace function_local_static {
// Same rule applies to a static local of a function template.
template <class T> int f() {
  static const int e = 1;
  decltype(&e) p = &e;
  static_assert(is_same<decltype(p), const int *>::value, "");
  static_assert(sizeof(p) == sizeof(const int *), "");
  return *p;
}
int use = f<int>() + f<char>();
} // namespace function_local_static
