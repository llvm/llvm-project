// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20                                         -verify=ref,both      %s

/// Global arrays with a zero filler only store the explicit elements and
/// one filler element.

constexpr int A[100] = {1, 2};
static_assert(A[0] == 1);
static_assert(A[1] == 2);
static_assert(A[2] == 0);
static_assert(A[99] == 0);
static_assert(*(A + 77) == 0);
static_assert(&A[50] - &A[0] == 50);
static_assert(&A[50] != &A[51]);
static_assert(&A[50] < &A[51]);
static_assert(A + 100 == &A[99] + 1);
static_assert(A[100] == 0); // both-error {{not an integral constant expression}} \
                            // both-note {{read of dereferenced one-past-the-end pointer}}

constexpr int sum(const int *P, int N) {
  int R = 0;
  for (int I = 0; I != N; ++I)
    R += P[I];
  return R;
}
static_assert(sum(A, 100) == 3);

constexpr int Z[100] = {};
static_assert(Z[0] == 0);
static_assert(Z[99] == 0);

constexpr char Str[100] = {'a', 'b'};
static_assert(__builtin_strlen(Str) == 2);
static_assert(__builtin_strlen(Str + 50) == 0);

constexpr double D[10] = {1.0};
static_assert(D[0] == 1.0);
static_assert(D[5] == 0.0);

constexpr int X = 0;
constexpr const int *P[10] = {&X};
static_assert(P[0] == &X);
static_assert(P[5] == nullptr);

struct S {
  int a;
  int b;
};
constexpr S SA[100] = {{1, 2}};
static_assert(SA[0].a == 1);
static_assert(SA[1].b == 0);
static_assert(SA[50].a == 0);
static_assert(&SA[50] - &SA[0] == 50);
static_assert(&SA[50].b == &SA[50].b);
static_assert(&SA[50].b != &SA[51].b);
constexpr int S::*MP = &S::b;
static_assert(SA[60].*MP == 0);
constexpr const S *SP = &SA[70];
static_assert(SP->b == 0);
static_assert(SP - SA == 70);

constexpr int M[10][10] = {{1}, {2}};
static_assert(M[0][0] == 1);
static_assert(M[1][0] == 2);
static_assert(M[5][5] == 0);
static_assert(&M[5][5] - &M[5][0] == 5);

struct T {
  int a = 0;
  S s[3];
};
constexpr T TA[10] = {{1, {{1, 2}}}};
static_assert(TA[0].s[0].b == 2);
static_assert(TA[5].s[2].b == 0);

namespace Ctor {
  struct S {
    int a;
    constexpr S() : a(12) {}
    constexpr S(int a) : a(a) {}
  };

  constexpr S foo[100] = {{13}};
  static_assert(foo[0].a == 13);
  static_assert(foo[1].a == 12);
  static_assert(foo[99].a == 12);
}

namespace StringLiterals {
  constexpr char s1[4] = "abc";
  static_assert(s1[0] == 'a');
  static_assert(s1[1] == 'b');
  static_assert(s1[2] == 'c');
  static_assert(s1[3] == '\0');

  constexpr char s2[100] = "abc";
  static_assert(s2[0] == 'a');
  static_assert(s2[1] == 'b');
  static_assert(s2[2] == 'c');
  static_assert(s2[3] == '\0');
  static_assert(s2[99] == '\0');

  constexpr char s3[100] = {"abc"};
  static_assert(s3[0] == 'a');
  static_assert(s3[1] == 'b');
  static_assert(s3[2] == 'c');
  static_assert(s3[3] == '\0');
  static_assert(s3[99] == '\0');
}

namespace NoFiller {
  struct A {
    constexpr A() : p(&p) {}
    void *p;
  };
  constexpr A a[10];
  static_assert(a[0].p == &a[0].p, "");
  static_assert(a[9].p == &a[9].p, "");
  static_assert(a[0].p != &a[9].p, "");
  static_assert(a[9].p != &a[0].p, "");

  constexpr A b[10] = {};
  static_assert(b[0].p == &b[0].p, "");
  static_assert(b[9].p == &b[9].p, "");
  static_assert(b[0].p != &b[9].p, "");
  static_assert(b[9].p != &b[0].p, "");
}

namespace NoFiller2 {
  struct A {
    constexpr A() : p(&p){}
    void*p;
  };
  struct W {
    A arr[2];
  };
  constexpr W w[10] = { {} };

  static_assert(w[5].arr[0].p == &w[5].arr[0].p);
}

namespace Dtor {
  constexpr int F[5] = {1,2};

  using T = int;
  constexpr int test1() { // both-error {{never produces a constant expression}}
    F[4].~T(); // both-note 2{{a constant expression cannot modify an object that is visible outside that expression}}
    return 1;
  }
  static_assert(test1() == 1); // both-error {{not an integral constant expression}} \
                               // both-note {{in call to}}

  struct S {
    constexpr ~S() {}
  };
  constexpr S FS[5] = {};
  constexpr int test2() {// both-error {{never produces a constant expression}}
    FS[4].~S();// both-note 2{{a constant expression cannot modify an object that is visible outside that expression}}
    return 1;
  }
  static_assert(test2() == 1);// both-error {{not an integral constant expression}} \
                               // both-note {{in call to}}
}
