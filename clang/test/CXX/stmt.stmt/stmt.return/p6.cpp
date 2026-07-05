// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s

auto&& f1() {
  return 42; // expected-error{{returning reference to local temporary object}}
}
const double& f2() {
  static int x = 42;
  return x; // expected-error{{returning reference to local temporary object}}
}
auto&& id(auto&& r) {
  return static_cast<decltype(r)&&>(r);
}
auto&& f3() {
  return id(42);        // OK, but probably a bug
}

void unevaluated() {
  using a = decltype ([] () -> const int & {
    const int &i = 0; // expected-note {{binding reference variable 'i' here}}
    return i; // expected-error{{returning reference to local temporary object}}
} ());
}

static_assert(__is_convertible(int, const int &));
static_assert(__is_nothrow_convertible(int, const int &));
