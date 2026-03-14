// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14
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
#if __has_feature(cxx_trailing_return)
  auto a = 0,
       (*b)() -> void, // expected-error {{declaration with trailing return type must be the only declaration in its group}}
       c = 0;
  auto d = 0,
       e() -> void, // expected-error {{declaration with trailing return type must be the only declaration in its group}}
       f = 0.0;
  auto x() -> void, // expected-error {{declaration with trailing return type must be the only declaration in its group}}
       y() -> void;
#endif

#if __has_feature(cxx_decltype)
  auto g = 0ull, h = decltype(g)(0);
#endif
}

#if __has_feature(cxx_trailing_return)
int F();
auto p = 0, (*q)() -> auto = F; // expected-error {{declaration with trailing return type must be the only declaration in its group}}
  #if __cplusplus < 201402L
  // expected-error@-2 {{'auto' not allowed in function return type}}
  #endif
#endif

#if __cplusplus >= 201402L
namespace DeducedReturnType {
  auto a = 0,
       b(), // expected-error {{function with deduced return type must be the only declaration in its group}}
       c = 0.0;
  auto d(), // expected-error {{function with deduced return type must be the only declaration in its group}}
       e = 1;
  auto f(), // expected-error {{function with deduced return type must be the only declaration in its group}}
       g();
}
#endif

template<typename T> void h() {
  auto a = T(), *b = &a;
#if __has_feature(cxx_decltype)
  auto c = T(), d = decltype(c)(0);
#endif
}
template void h<int>();
template void h<unsigned long>();
