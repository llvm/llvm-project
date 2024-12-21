// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -fcxx-exceptions -triple=x86_64-linux-gnu
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s -fcxx-exceptions -triple=x86_64-linux-gnu

using size_t = decltype(sizeof(int));

int &operator ""_x1 (const char *);
double &operator ""_x1 (const char *, size_t);
double &i1 = "foo"_x1;
#if __cplusplus >= 202002L
using char8 = float;
float &operator ""_x1 (const char8_t *, size_t);
#else
using char8 = double;
#endif
char8 &i2 = u8"foo"_x1;
double &i3 = L"foo"_x1; // expected-error {{no matching literal operator for call to 'operator""_x1' with arguments of types 'const wchar_t *' and 'unsigned long'}}

char &operator ""_x1(const wchar_t *, size_t);
char &i4 = L"foo"_x1; // ok
double &i5 = R"(foo)"_x1; // ok
char8 &i6 = u\
8\
R\
"(foo)"\
_\
x\
1; // ok

#if __cplusplus >= 202002L
template<int N> struct S {
  char a[N];
  constexpr S(const char (&r)[N]) {
    __builtin_memcpy(a, r, N);
    if (a[0] == 'x') throw "no";
  }
  constexpr ~S() {
    if (a[0] == 'y') throw "also no";
  }
};

// Check the produced contents are correct.
template<S s> constexpr const decltype(s) &operator""_str() { return s; }
static_assert(__builtin_strcmp("hello world"_str.a, "hello world") == 0);

template<S> float &operator""_s();
void no_fallback() {
  "hello"_s;
  // FIXME: It'd be useful to explain what candidates were found and why they didn't work.
  "xyzzy"_s; // expected-error {{no matching literal operator for call to 'operator""_s' with arguments of types 'const char *' and 'unsigned long', and no matching literal operator template}}
  "yello"_s; // expected-error {{no matching literal operator for call to 'operator""_s' with arguments of types 'const char *' and 'unsigned long', and no matching literal operator template}}
}

double &operator""_s(const char*, size_t);
void f() {
  float &a = "foo"_s;
  double &b = "xar"_s;
  double &c = "yar"_s;
}

template<S<4>> float &operator""_t();
double &operator""_t(const char*, size_t);
void g() {
  double &a = "fo"_t;
  float &b = "foo"_t;
  double &c = "fooo"_t;
}

template<int N> struct X {
  static constexpr int size = N;
  constexpr X(const char (&r)[N]) {}
};
template<X x> requires (x.size == 4) // expected-note {{because 'X<5>{}.size == 4' (5 == 4) evaluated to false}}
void operator""_x(); // expected-note {{constraints not satisfied}}
void operator""_x(const char*, size_t) = delete;

template<int N> requires (N == 4)
struct Y {
  constexpr Y(const char (&r)[N]) {}
};
template<Y> float &operator""_y();
void operator""_y(const char*, size_t) = delete; // expected-note {{deleted here}}

void test() {
  "foo"_x;
  "foo"_y;

  // We only check the template argument itself for validity, not the whole
  // call, when deciding whether to use the template or non-template form.
  "fooo"_x; // expected-error {{no matching function}}
  "fooo"_y; // expected-error {{deleted function}}
}
#endif
