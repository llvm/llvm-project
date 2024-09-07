// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct S0 {
  int x;
  static const int test0 = __alignof__(x); // expected-error {{invalid application of 'alignof' to a field of a class still being defined}}
  static const int test1 = __alignof__(S0::x); // expected-error {{invalid application of 'alignof' to a field of a class still being defined}}
  auto test2() -> char(&)[__alignof__(x)]; // expected-error {{invalid application of 'alignof' to a field of a class still being defined}}
};

struct S1; // expected-note 6 {{forward declaration}}
extern S1 s1;
const int test3 = __alignof__(s1); // expected-error {{invalid application of '__alignof' to an incomplete type 'S1'}}

struct S2 {
  S2();
  S1 &s;
  int x;

  int test4 = __alignof__(x); // ok
  int test5 = __alignof__(s); // expected-error {{invalid application of '__alignof' to an incomplete type 'S1'}}
};

const int test6 = __alignof__(S2::x);
const int test7 = __alignof__(S2::s); // expected-error {{invalid application of '__alignof' to an incomplete type 'S1'}}

// Arguably, these should fail like the S1 cases do: the alignment of
// 's2.x' should depend on the alignment of both x-within-S2 and
// s2-within-S3 and thus require 'S3' to be complete.  If we start
// doing the appropriate recursive walk to do that, we should make
// sure that these cases don't explode.
struct S3 {
  S2 s2;

  static const int test8 = __alignof__(s2.x);
  static const int test9 = __alignof__(s2.s); // expected-error {{invalid application of '__alignof' to an incomplete type 'S1'}}
  auto test10() -> char(&)[__alignof__(s2.x)];
  static const int test11 = __alignof__(S3::s2.x);
  static const int test12 = __alignof__(S3::s2.s); // expected-error {{invalid application of '__alignof' to an incomplete type 'S1'}}
  auto test13() -> char(&)[__alignof__(s2.x)];
};

// Same reasoning as S3.
struct S4 {
  union {
    int x;
  };
  static const int test0 = __alignof__(x);
  static const int test1 = __alignof__(S0::x);
  auto test2() -> char(&)[__alignof__(x)];
};

// Regression test for asking for the alignment of a field within an invalid
// record.
struct S5 {
  S1 s;  // expected-error {{incomplete type}}
  int x;
};
const int test8 = __alignof__(S5::x);

int test14[2];

static_assert(alignof(test14) == 4, "foo"); // expected-warning {{'alignof' applied to an expression is a GNU extension}}

// PR19992
static_assert(alignof(int[]) == alignof(int), ""); // ok

namespace alignof_array_expr {
  alignas(32) extern int n[];
  static_assert(alignof(n) == 32, ""); // expected-warning {{GNU extension}}

  template<int> struct S {
    static int a[];
  };
  template<int N> int S<N>::a[N];
  // ok, does not complete type of S<-1>::a
  static_assert(alignof(S<-1>::a) == alignof(int), ""); // expected-warning {{GNU extension}}
}

template <typename T> void n(T) {
  alignas(T) int T1;
  char k[__alignof__(T1)];
  static_assert(sizeof(k) == alignof(long long), "");
}
template void n(long long);

namespace PR22042 {
template <typename T>
void Fun(T A) {
  typedef int __attribute__((__aligned__(A))) T1; // expected-error {{requested alignment is dependent but declaration is not dependent}}
  int k1[__alignof__(T1)];
}

template <int N>
struct S {
  typedef __attribute__((aligned(N))) int Field[sizeof(N)]; // expected-error {{requested alignment is dependent but declaration is not dependent}}
};
}

typedef int __attribute__((aligned(16))) aligned_int;
template <typename>
using template_alias = aligned_int;
static_assert(alignof(template_alias<void>) == 16, "Expected alignment of 16" );

struct PR47138 {
  invalid_type a; // expected-error {{unknown type}}
};
static_assert(__alignof__(PR47138) == 1, ""); // Don't crash.
