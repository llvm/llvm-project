// RUN: %clang_cc1 -std=c++98 -verify=expected %s
// RUN: %clang_cc1 -std=c++11 -verify=expected %s
// RUN: %clang_cc1 -std=c++14 -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -verify=expected,since-cxx20 %s
// RUN: %clang_cc1 -std=c++23 -verify=expected,since-cxx20,since-cxx23 %s
// RUN: %clang_cc1 -std=c++2c -verify=expected,since-cxx20,since-cxx23,since-cxx26 %s

#if __cplusplus < 202002L
// expected-no-diagnostics
#endif

namespace dr2847 { // dr2847: 19

#if __cplusplus >= 202002L

template<typename>
void i();

struct A {
  template<typename>
  void f() requires true;

  template<>
  void f<int>() requires true;
  // since-cxx20-error@-1 {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

  friend void i<int>() requires true;
  // since-cxx20-error@-1 {{friend specialization cannot have a trailing requires clause unless it declares a function template}}
};

template<typename>
struct B {
  void f() requires true;

  template<typename>
  void g() requires true;

  template<typename>
  void h() requires true;

  template<>
  void h<int>() requires true;
  // since-cxx20-error@-1 {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

  friend void i<int>() requires true;
  // since-cxx20-error@-1 {{friend specialization cannot have a trailing requires clause unless it declares a function template}}
};

template<>
void B<int>::f() requires true;
// since-cxx20-error@-1 {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

template<>
template<typename T>
void B<int>::g() requires true;

#endif

} // namespace dr2847

namespace dr2858 { // dr2858: 19

#if __cplusplus > 202302L

template<typename... Ts>
struct A {
  // FIXME: The nested-name-specifier in the following friend declarations are declarative,
  // but we don't treat them as such (yet).
  friend void Ts...[0]::f();
  template<typename U>
  friend void Ts...[0]::g();

  friend struct Ts...[0]::B;
  // FIXME: The index of the pack-index-specifier is printed as a memory address in the diagnostic.
  template<typename U>
  friend struct Ts...[0]::C;
  // expected-warning-re@-1 {{dependent nested name specifier 'Ts...[{{.*}}]::' for friend template declaration is not supported; ignoring this friend declaration}}
};

#endif

} // namespace dr2858
