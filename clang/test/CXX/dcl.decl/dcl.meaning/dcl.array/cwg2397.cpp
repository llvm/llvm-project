// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14

void f() {
  int a[5];
  auto (*b)[5] = &a;
  auto (&c)[5] = a;
  auto (&&d)[5] = static_cast<int(&&)[5]>(a);
  auto e[] = {0}; // expected-error{{cannot deduce actual type for variable 'e' with type 'auto[]' from initializer list}}
  static_assert(__is_same(decltype(b), int (*)[5]), "");
  static_assert(__is_same(decltype(c), int (&)[5]), "");
  static_assert(__is_same(decltype(d), int (&&)[5]), "");
}

#if __cplusplus >= 201402L

constexpr int g() {
  int a[] = {1, 2, 3};
  auto (&b)[3] = a;
  return b[1];
}

static_assert(g() == 2, "");

#endif
