// RUN: %clang_cc1 -std=c++17 %s -verify
// RUN: %clang_cc1 -std=c++20 %s -verify
// RUN: %clang_cc1 -std=c++20 %s -verify -Wc++17-compat -DCOMPAT
//
// RUN: %clang_cc1 -std=c++17 %s -E -o - | FileCheck %s --check-prefix=CXX17
// RUN: %clang_cc1 -std=c++20 %s -E -o - | FileCheck %s --check-prefix=CXX20

namespace N {

struct A {};
void operator<=(A, A);
#if __cplusplus > 201703L
void operator<=>(A, A);
#ifdef COMPAT
// expected-warning@-2 {{'<=>' operator is incompatible with C++ standards before C++20}}
#endif
#endif

template<auto> struct X {};
X<operator<=>
#if __cplusplus <= 201703L
  // expected-warning@-2 {{'<=>' is a single token in C++20; add a space to avoid a change in behavior}}
#else
  >
#endif
#ifdef COMPAT
// expected-warning@-7 {{'<=>' operator is incompatible with C++ standards before C++20}}
#endif
  x;
}

// <=> can be formed by pasting other comparison operators.
#if __cplusplus > 201703L
#define STR(x) #x
#define STR_EXPANDED(x) STR(x)
#define PASTE(x, y) x ## y
constexpr char a[] = STR_EXPANDED(PASTE(<, =>));
constexpr char b[] = STR_EXPANDED(PASTE(<=, >));
static_assert(__builtin_strcmp(a, "<=>") == 0);
static_assert(__builtin_strcmp(b, "<=>") == 0);
#endif

// -E must not accidentally form a <=> token.

// CXX17: preprocess1: < =>
// CXX17: preprocess2: <=>
// CXX17: preprocess3: < =>
// CXX17: preprocess4: <=>=
// CXX17: preprocess5: <=>>
// CXX17: preprocess6: <=>>=
// CXX17: preprocess7: <=>
// CXX17: preprocess8: <=>=
//
// CXX20: preprocess1: < =>
// CXX20: preprocess2: <= >
// CXX20: preprocess3: < =>
// CXX20: preprocess4: <= >=
// CXX20: preprocess5: <= >>
// CXX20: preprocess6: <= >>=
// CXX20: preprocess7: <=>
// CXX20: preprocess8: <=>=

#define ID(x) x
[[some_vendor::some_attribute( // expected-warning {{unknown attribute}}
preprocess1: ID(<)ID(=>),
preprocess2: ID(<=)ID(>),
preprocess3: ID(<)ID(=)ID(>),
preprocess4: ID(<=)ID(>=),
preprocess5: ID(<=)ID(>>),
preprocess6: ID(<=)ID(>>=),
preprocess7: ID(<=>) // expected-warning 0-1{{'<=>'}}
preprocess8: ID(<=>=) // expected-warning 0-1{{'<=>'}}
)]];
