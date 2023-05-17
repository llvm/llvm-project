// RUN: %clang_cc1 %s -std=c++17 -fsyntax-only -fcxx-exceptions -verify

struct S1 {
  void f() [[clang::annotate_type("foo")]];
  [[clang::annotate_type("foo")]] void g(); // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
};

template <typename T1, typename T2> struct is_same {
  static constexpr bool value = false;
};

template <typename T1> struct is_same<T1, T1> {
  static constexpr bool value = true;
};

static_assert(is_same<int, int [[clang::annotate_type("foo")]]>::value);
static_assert(is_same<int [[clang::annotate_type("foo")]],
                      int [[clang::annotate_type("bar")]]>::value);
static_assert(is_same<int *, int *[[clang::annotate_type("foo")]]>::value);

// Cannot overload on types that only differ by `annotate_type` attribute.
void f(int) {} // expected-note {{previous definition is here}}
void f(int [[clang::annotate_type("foo")]]) {} // expected-error {{redefinition of 'f'}}

// Cannot specialize on types that only differ by `annotate_type` attribute.
template <class T> struct S2 {};

template <> struct S2<int> {}; // expected-note {{previous definition is here}}

template <>
struct S2<int [[clang::annotate_type("foo")]]> {}; // expected-error {{redefinition of 'S2<int>'}}

// Test that the attribute supports parameter pack expansion.
template <int... Is> void variadic_func_template() {
  int [[clang::annotate_type("foo", Is...)]] val;
}
int f2() { variadic_func_template<1, 2, 3>(); }

// Make sure we correctly diagnose wrong number of arguments for
// [[clang::annotate_type]] inside a template argument.
template <typename Ty> void func_template();
void f3() {
  func_template<int [[clang::annotate_type()]]>(); // expected-error {{'annotate_type' attribute takes at least 1 argument}}
}

// More error cases: Prohibit adding the attribute to declarations.
// Different declarations hit different code paths, so they need separate tests.
namespace [[clang::annotate_type("foo")]] my_namespace {} // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
struct [[clang::annotate_type("foo")]] S3; // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
struct [[clang::annotate_type("foo")]] S3{ // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
  [[clang::annotate_type("foo")]] int member; // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
};
void f4() {
  for ([[clang::annotate_type("foo")]] int i = 0; i < 42; ++i) {} // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
  for (; [[clang::annotate_type("foo")]] bool b = false;) {} // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
  while ([[clang::annotate_type("foo")]] bool b = false) {} // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
  if ([[clang::annotate_type("foo")]] bool b = false) {} // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
  try {
  } catch ([[clang::annotate_type("foo")]] int i) { // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
  }
}
template <class T>
[[clang::annotate_type("foo")]] T var_template; // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
[[clang::annotate_type("foo")]] extern "C" int extern_c_func(); // expected-error {{an attribute list cannot appear here}}
extern "C" [[clang::annotate_type("foo")]] int extern_c_func(); // expected-error {{'annotate_type' attribute cannot be applied to a declaration}}
