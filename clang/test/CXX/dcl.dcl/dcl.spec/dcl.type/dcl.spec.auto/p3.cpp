// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98 -Wno-c++11-extensions -Wc++11-compat 
void f() {
  auto a = a; // expected-error{{variable 'a' declared with deduced type 'auto' cannot appear in its own initializer}}
  auto *b = b; // expected-error{{variable 'b' declared with deduced type 'auto *' cannot appear in its own initializer}}
  const auto c = c; // expected-error{{variable 'c' declared with deduced type 'const auto' cannot appear in its own initializer}}
  if (auto d = d) {} // expected-error {{variable 'd' declared with deduced type 'auto' cannot appear in its own initializer}}
  auto e = ({ auto f = e; 0; }); // expected-error {{variable 'e' declared with deduced type 'auto' cannot appear in its own initializer}}
}

void g() {
  auto a; // expected-error{{declaration of variable 'a' with deduced type 'auto' requires an initializer}}
  
  auto *b; // expected-error{{declaration of variable 'b' with deduced type 'auto *' requires an initializer}}

  if (auto b) {} // expected-error {{must have an initializer}}
  for (;auto b;) {} // expected-error {{must have an initializer}}
  while (auto b) {} // expected-error {{must have an initializer}}
  if (auto b = true) { (void)b; }
}

auto n(1,2,3); // expected-error{{initializer for variable 'n' with type 'auto' contains multiple expressions}}

namespace N
{
  auto a = "const char [16]", *p = &a;
}

void h() {
  auto b = 42ULL;

  for (auto c = 0; c < b; ++c) {
  }
}

template<typename T, typename U> struct same;
template<typename T> struct same<T, T> {};

void p3example() {
  auto x = 5;
  const auto *v = &x, u = 6;
  static auto y = 0.0;
  // In C++98: 'auto' storage class specifier is redundant and incompatible with C++0x
  // In C++0x: 'auto' storage class specifier is not permitted in C++0x, and will not be supported in future releases
  auto int r; // expected-warning {{'auto' storage class specifier}}

  same<__typeof(x), int> xHasTypeInt;
  same<__typeof(v), const int*> vHasTypeConstIntPtr;
  same<__typeof(u), const int> uHasTypeConstInt;
  same<__typeof(y), double> yHasTypeDouble;
}

#if __cplusplus >= 201103L
namespace PR13293 {
  // Ensure that dependent declarators have their deduction delayed.
  int f(char);
  double f(short);
  template<typename T> struct S {
    static constexpr auto (*p)(T) = &f;
  };

  constexpr int (*f1)(char) = &f;
  constexpr double (*f2)(short) = &f;
  static_assert(S<char>::p == f1, "");
  static_assert(S<short>::p == f2, "");

  struct K { int n; };
  template<typename T> struct U {
    static constexpr auto (T::*p) = &K::n;
  };
  static_assert(U<K>::p == &K::n, "");

  template<typename T>
  using X = auto(int) -> auto(*)(T) -> auto(*)(char) -> long;
  X<double> x;
  template<typename T> struct V {
    //static constexpr auto (*p)(int) -> auto(*)(T) -> auto(*)(char) = &x; // ill-formed
    static constexpr auto (*(*(*p)(int))(T))(char) = &x; // ok
  };
  V<double> v;

  int *g(double);
  template<typename T> void h() {
    new (auto(*)(T)) (&g);
  }
  template void h<double>();
}
#endif

auto fail((unknown)); // expected-error{{use of undeclared identifier 'unknown'}}
int& crash = fail;
