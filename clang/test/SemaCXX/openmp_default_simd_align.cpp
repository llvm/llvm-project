// RUN: %clang_cc1 -std=c++11 -fsyntax-only -triple x86_64-unknown-unknown -verify %s

struct S0 {
  int x;
  static const int test0 = __builtin_omp_required_simd_align(x); // expected-error {{invalid application of '__builtin_omp_required_simd_align' to an expression, only type is allowed}}
  static const int test1 = __builtin_omp_required_simd_align(decltype(S0::x));
  auto test2() -> char(&)[__builtin_omp_required_simd_align(decltype(x))];
};

struct S1; // expected-note 6 {{forward declaration}}
extern S1 s1;
const int test3 = __builtin_omp_required_simd_align(decltype(s1)); // expected-error {{invalid application of '__builtin_omp_required_simd_align' to an incomplete type 'decltype(s1)' (aka 'S1')}}

struct S2 {
  S2();
  S1 &s;
  int x;

  int test4 = __builtin_omp_required_simd_align(decltype(x)); // ok
  int test5 = __builtin_omp_required_simd_align(decltype(s)); // expected-error {{invalid application of '__builtin_omp_required_simd_align' to an incomplete type 'S1'}}
};

const int test6 = __builtin_omp_required_simd_align(decltype(S2::x));
const int test7 = __builtin_omp_required_simd_align(decltype(S2::s)); // expected-error {{invalid application of '__builtin_omp_required_simd_align' to an incomplete type 'S1'}}

// Arguably, these should fail like the S1 cases do: the alignment of
// 's2.x' should depend on the alignment of both x-within-S2 and
// s2-within-S3 and thus require 'S3' to be complete.  If we start
// doing the appropriate recursive walk to do that, we should make
// sure that these cases don't explode.
struct S3 {
  S2 s2;

  static const int test8 = __builtin_omp_required_simd_align(decltype(s2.x));
  static const int test9 = __builtin_omp_required_simd_align(decltype(s2.s)); // expected-error {{invalid application of '__builtin_omp_required_simd_align' to an incomplete type 'S1'}}
  auto test10() -> char(&)[__builtin_omp_required_simd_align(decltype(s2.x))];
  static const int test11 = __builtin_omp_required_simd_align(decltype(S3::s2.x));
  static const int test12 = __builtin_omp_required_simd_align(decltype(S3::s2.s)); // expected-error {{invalid application of '__builtin_omp_required_simd_align' to an incomplete type 'S1'}}
  auto test13() -> char(&)[__builtin_omp_required_simd_align(decltype(s2.x))];
};

// Same reasoning as S3.
struct S4 {
  union {
      int x;
    };
  static const int test0 = __builtin_omp_required_simd_align(decltype(x));
  static const int test1 = __builtin_omp_required_simd_align(decltype(S0::x));
  auto test2() -> char(&)[__builtin_omp_required_simd_align(decltype(x))];
};

// Regression test for asking for the alignment of a field within an invalid
// record.
struct S5 {
  S1 s;  // expected-error {{incomplete type}}
  int x;
};
const int test8 = __builtin_omp_required_simd_align(decltype(S5::x));

long long int test14[2];

static_assert(__builtin_omp_required_simd_align(decltype(test14)) == 16, "foo");

static_assert(__builtin_omp_required_simd_align(int[2]) == __builtin_omp_required_simd_align(int), ""); // ok

namespace __builtin_omp_required_simd_align_array_expr {
  alignas(32) extern int n[2];
  static_assert(__builtin_omp_required_simd_align(decltype(n)) == 16, "");

  template<int> struct S {
      static int a[];
    };
  template<int N> int S<N>::a[N];
  static_assert(__builtin_omp_required_simd_align(decltype(S<1>::a)) == __builtin_omp_required_simd_align(int), "");
  static_assert(__builtin_omp_required_simd_align(decltype(S<1128>::a)) == __builtin_omp_required_simd_align(int), "");
}

template <typename T> void n(T) {
  alignas(T) int T1;
  char k[__builtin_omp_required_simd_align(decltype(T1))];
  static_assert(sizeof(k) == __builtin_omp_required_simd_align(long long), "");
}
template void n(long long);
