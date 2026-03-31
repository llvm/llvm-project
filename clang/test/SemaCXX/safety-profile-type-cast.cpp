// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++20 %s
// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::type_cast)]];

void test_violation() {
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]]
void test_suppress_decl() {
  int *p = reinterpret_cast<int*>(0);
}

void test_suppress_stmt() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] {
    int *p = reinterpret_cast<int*>(0);
  }
  int *q = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

void test_suppress_stmt_with_rule() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast, rule: "reinterpret_cast")]] {
    int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  }
}

// P3589R2 [decl.attr.enforce]p2: static semantics applied after translation
// phase 7 -- no diagnostic in template definition, only at instantiation.
template <typename T>
void template_cast(T x) {
  auto *p = reinterpret_cast<int*>(x); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}
void instantiate() {
  template_cast(0); // expected-note {{in instantiation of function template specialization 'template_cast<int>' requested here}}
}

// P3589R2 Section 1.1: profile violations must not affect overload resolution.
// The reinterpret_cast in the decltype return type is in an unevaluated
// context, so the profile check is suppressed and doesn't SFINAE out the
// first overload. The selected overload's body has no violation.
template <typename T>
auto sfinae_cast(T x) -> decltype(reinterpret_cast<int*>(x)) {
  return nullptr;
}
template <typename T>
int sfinae_cast(...) { return 0; }
void test_sfinae() {
  int *p = sfinae_cast<long>(0L);
}

// Profile violations are suppressed in unevaluated contexts.
void test_unevaluated() {
  using T = decltype(reinterpret_cast<int*>(0));
}
