// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Wc++11-compat -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++1z -Wc++11-compat -verify %s

#if __cplusplus < 201103L

namespace N {
  template<typename T> void f(T) {} // expected-note 2{{here}}
  namespace M {
    template void ::N::f<int>(int); // expected-warning {{explicit instantiation of 'f' not in a namespace enclosing 'N'}}
  }
}
using namespace N;
template void f<char>(char); // expected-warning {{explicit instantiation of 'N::f' must occur in namespace 'N'}}

template<typename T> void g(T) {} // expected-note 2{{here}}
namespace M {
  template void g<int>(int); // expected-warning {{explicit instantiation of 'g' must occur at global scope}}
  template void ::g<char>(char); // expected-warning {{explicit instantiation of 'g' must occur at global scope}}
}

template inline void g<double>(double); // expected-warning {{explicit instantiation cannot be 'inline'}}

void g() {
  auto int n = 0; // expected-warning {{'auto' storage class specifier is redundant and incompatible with C++11}}
}

int n;
struct S {
  char c;
}
s = { n }, // expected-warning {{non-constant-expression cannot be narrowed from type 'int' to 'char' in initializer list in C++11}} expected-note {{explicit cast}}
t = { 1234 }; // expected-warning {{constant expression evaluates to 1234 which cannot be narrowed to type 'char' in C++11}} expected-warning {{changes value}} expected-note {{explicit cast}}

#define PRIuS "uS"
int printf(const char *, ...);
typedef __typeof(sizeof(int)) size_t;
void h(size_t foo, size_t bar) {
  printf("foo is %"PRIuS", bar is %"PRIuS, foo, bar); // expected-warning 2{{identifier after literal will be treated as a reserved user-defined literal suffix in C++11}}
}

#define _x + 1
char c = 'x'_x; // expected-warning {{will be treated as a user-defined literal suffix}}

template<int ...N> int f() { // expected-warning {{C++11 extension}}
  return (N + ...); // expected-warning {{C++1z extension}}
}

#else

auto init_capture = [a(0)] {}; // expected-warning {{initialized lambda captures are incompatible with C++ standards before C++14}}
static_assert(true); // expected-warning {{incompatible with C++ standards before C++1z}}

template<int ...N> int f() { return (N + ...); } // expected-warning {{incompatible with C++ standards before C++1z}}

#endif
