// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98 -Wno-c++11-extensions
void f() {
  auto a = 0, b = 0, c = 0;
  auto d = 0, e = 0.0; // expected-error {{'int' in declaration of 'd' and deduced as 'double' in declaration of 'e'}}

  auto v1 = 0, *p1 = &v1;
  auto *p2 = 0, v2 = *p2; // expected-error {{incompatible initializer}}

  const int k = 0;
  auto &f = k, &g = a; // expected-error {{'const int' in declaration of 'f' and deduced as 'int' in declaration of 'g'}}

  typedef int I;
  I x;
  auto xa = x, xb = 0;

  auto &&ra1 = a, rb1 = b; // expected-error {{'int &' in declaration of 'ra1' and deduced as 'int' in declaration of 'rb1'}}
  auto &&ra2 = +a, rb2 = b;
}

void g() {
  auto a = 0,
#if __has_feature(cxx_trailing_return)
       (*b)() -> void,
#endif
       c = 0;
  auto d = 0, // expected-error {{'auto' deduced as 'int' in declaration of 'd' and deduced as 'double' in declaration of 'f'}}
#if __has_feature(cxx_trailing_return)
       (*e)() -> void,
#endif
       f = 0.0;
}
