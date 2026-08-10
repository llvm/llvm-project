// RUN: %clang_cc1 -std=c++23 -fsyntax-only -Wno-unused %s -verify=cxx20-cxx26
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -Wno-unused %s -verify=cxx20,cxx20-cxx26
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -Wno-unused %s -verify=precxx20,cxx11-cxx17
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -Wno-unused %s -verify=precxx20,cxx11-cxx17
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wno-unused %s -verify=precxx20,cxx11-cxx17
// RUN: %clang_cc1 -std=c++03 -fsyntax-only -Wno-unused %s -verify=precxx20
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -Wno-unused %s -verify=precxx20
// RUN: %clang_cc1 -x c -std=c23 -fsyntax-only -Wno-unused %s -verify=c

#if __has_builtin(__builtin_is_within_lifetime)
#error has the builtin
#else
#error does not have the builtin
#endif
// cxx20-cxx26-error@-4 {{has the builtin}}
// precxx20-error@-3 {{does not have the builtin}}
// c-error@-4 {{does not have the builtin}}

#if __has_constexpr_builtin(__builtin_is_within_lifetime)
#error has the constexpr builtin
#else
#error does not have the constexpr builtin
#endif
// cxx20-cxx26-error@-4 {{has the constexpr builtin}}
// precxx20-error@-3 {{does not have the constexpr builtin}}
// c-error@-4 {{does not have the constexpr builtin}}

#if __cplusplus < 201103L
#define static_assert __extension__ _Static_assert
#define CONSTEXPR11
#else
#define CONSTEXPR11 constexpr
#endif

static const int i1 = 0;
static_assert(__builtin_is_within_lifetime(&i1), "");
// precxx20-error@-1 {{use of undeclared identifier '__builtin_is_within_lifetime'}}
// c-error@-2 {{use of undeclared identifier '__builtin_is_within_lifetime'}}

#if !defined(__cplusplus) || __cplusplus >= 201102L
constexpr int i2 = 0;
static_assert(__builtin_is_within_lifetime(&i2), "");
// cxx11-cxx17-error@-1 {{use of undeclared identifier '__builtin_is_within_lifetime'}}
// c-error@-2 {{use of undeclared identifier '__builtin_is_within_lifetime'}}
#endif

#ifdef __cplusplus
template<typename T>
CONSTEXPR11 bool f1(T i) {  // #f1
  return __builtin_is_within_lifetime(&i);  // #f1-consteval-call
}

bool(&fp1)(int) = f1<int>;
// cxx20-cxx26-error@-1 {{cannot take address of immediate function 'f1<int>' outside of an immediate invocation}}
//   cxx20-cxx26-note@#f1 {{declared here}}
//   cxx20-cxx26-note@#f1-consteval-call {{'f1<int>' is an immediate function because its body contains a call to a consteval function '__builtin_is_within_lifetime' and that call is not a constant expression}}
// precxx20-error@#f1-consteval-call {{use of undeclared identifier '__builtin_is_within_lifetime'}}
//   precxx20-note@-5 {{in instantiation of function template specialization 'f1<int>' requested here}}
#else
void f1(int i) {
  __builtin_is_within_lifetime(&i);
  // c-error@-1 {{use of undeclared identifier '__builtin_is_within_lifetime'}}
}
#endif

#if __cplusplus >= 202002L
constexpr void f2() {
  int i = 0;
  if consteval {  // cxx20-warning {{consteval if}}
    __builtin_is_within_lifetime(&i);
  }
}
void(&fp2)() = f2;

constexpr void f3() {
  __builtin_is_within_lifetime(&i1);
}
void(&fp3)() = f3;

constexpr void f4() {
  &__builtin_is_within_lifetime;
  // cxx20-cxx26-error@-1 {{builtin functions must be directly called}}
  // cxx20-cxx26-error@-2 {{cannot take address of consteval function '__builtin_is_within_lifetime' outside of an immediate invocation}}
  __builtin_is_within_lifetime();
  // cxx20-cxx26-error@-1 {{too few arguments to function call, expected 1, have 0}}
  // cxx20-cxx26-error@-2 {{cannot take address of consteval function '__builtin_is_within_lifetime' outside of an immediate invocation}}
  int* not_constexpr;
  __builtin_is_within_lifetime(not_constexpr);
  // cxx20-cxx26-error@-1 {{call to consteval function '__builtin_is_within_lifetime' is not a constant expression}}
  //   cxx20-cxx26-note@-2 {{read of non-constexpr variable 'not_constexpr' is not allowed in a constant expression}}
  //   cxx20-cxx26-note@-4 {{declared here}}
}
#endif
