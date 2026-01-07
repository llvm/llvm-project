// RUN: %clang_cc1 -triple x86_64-linux -std=c23 -fsyntax-only -verify %s
// expected-no-diagnostics

#define INT64_MIN (-9223372036854775807LL - 1)

void t1() {
  {
    enum { A };
    enum { B };

    _Static_assert(_Generic(A, typeof(B): 1, default: 0) == 1, "");
    _Static_assert(_Generic(typeof(A), typeof(B): 1, default: 0) == 1, "");
  }

  {
    _Static_assert(
      _Generic(typeof(enum {A}), typeof(enum {B}): 1, default: 0) == 0, "");
  }
}

void t2() {
  {
    enum : int { A };
    enum : int { B };

    _Static_assert(_Generic(A, typeof(B): 1, default: 0) == 0, "");
    _Static_assert(_Generic(typeof(A), typeof(B): 1, default: 0) == 0, "");
  }

  {
    _Static_assert(
      _Generic(typeof(enum : int{A}), typeof(enum : int{B}): 1, default: 0) == 0, "");
  }
}

void t3() {
  {
    enum { A = INT64_MIN };
    enum { B = INT64_MIN };

    _Static_assert(_Generic(A, __typeof__(B): 1, default: 0) == 0, "");
    _Static_assert(_Generic(__typeof__(A), __typeof__(B): 1, default: 0) == 0, "");
  }

  {
    enum : long long { A = INT64_MIN };
    enum : long long { B = INT64_MIN };

    _Static_assert(_Generic(A, __typeof__(B): 1, default: 0) == 0, "");
    _Static_assert(_Generic(__typeof__(A), __typeof__(B): 1, default: 0) == 0, "");
  }
}

void t4() {
  enum : int { A };
  enum : int { B };

  _Static_assert(_Generic(A, typeof(B): 1, typeof(A): 2, default: 0) == 2, "");
}
