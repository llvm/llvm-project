// RUN: %clang_cc1 -DWIN -verify=nan-not-constant -std=c++23 -fsyntax-only  %s
// RUN: %clang_cc1 -verify -std=c++23 -fsyntax-only  %s
// RUN: %clang_cc1 -verify=cplusplus20andless -std=c++20 -fsyntax-only  %s

// expected-no-diagnostics


#ifdef WIN
#define INFINITY ((float)(1e+300 * 1e+300))
#define NAN      (-(float)(INFINITY * 0.0F))
#else
#define NAN (__builtin_nanf(""))
#define INFINITY (__builtin_inff())
#endif

template <auto, auto> constexpr bool is_same_val = false;
template <auto X> constexpr bool is_same_val<X, X> = true;

extern "C" {
  double fmin(double, double);
  float fminf(float, float);
  long double fminl(long double, long double);

  double fmax(double, double);
  float fmaxf(float, float);
  long double fmaxl(long double, long double);

  double frexp(double, int*);
  float frexpf(float, int*);
  long double frexpl(long double, int*);
}

#if __cplusplus <= 202002L
#define CALL_BUILTIN(BASE_NAME, ...) __builtin_##BASE_NAME(__VA_ARGS__)
#else
#define CALL_BUILTIN(BASE_NAME, ...) BASE_NAME(__VA_ARGS__)
#endif

int main() {
  int i;

  // fmin
  static_assert(is_same_val<CALL_BUILTIN(fmin, 15.24, 1.3), 1.3>);
  static_assert(is_same_val<CALL_BUILTIN(fmin, -0.0, +0.0), -0.0>);
  static_assert(is_same_val<CALL_BUILTIN(fmin, +0.0, -0.0), -0.0>);
  static_assert(is_same_val<CALL_BUILTIN(fminf, NAN, -1), -1.f>);
  // nan-not-constant-error@-1 {{non-type template argument is not a constant expression}}
  // nan-not-constant-note@-2 {{floating point arithmetic produces a NaN}}
  static_assert(is_same_val<CALL_BUILTIN(fminf, +INFINITY, 0), 0.f>);
  static_assert(is_same_val<CALL_BUILTIN(fminf, -INFINITY, 0), -INFINITY>);
  static_assert(is_same_val<CALL_BUILTIN(fminf, NAN, NAN), NAN>);
  // nan-not-constant-error@-1 {{non-type template argument is not a constant expression}}
  // nan-not-constant-note@-2 {{floating point arithmetic produces a NaN}}
  static_assert(is_same_val<CALL_BUILTIN(fminl, 123.456L, 789.012L), 123.456L>);

  static_assert(is_same_val<CALL_BUILTIN(fmax, 15.24, 1.3), 15.24>);
  static_assert(is_same_val<CALL_BUILTIN(fmax, -0.0, +0.0), +0.0>);
  static_assert(is_same_val<CALL_BUILTIN(fmax, +0.0, -0.0), +0.0>);
  static_assert(is_same_val<CALL_BUILTIN(fmaxf, NAN, -1), -1.f>);
  // nan-not-constant-error@-1 {{non-type template argument is not a constant expression}}
  // nan-not-constant-note@-2 {{floating point arithmetic produces a NaN}}
  static_assert(is_same_val<CALL_BUILTIN(fmaxf, +INFINITY, 0), INFINITY>);
  static_assert(is_same_val<CALL_BUILTIN(fmaxf, -INFINITY, 0), 0.f>);
  static_assert(is_same_val<CALL_BUILTIN(fmaxf, NAN, NAN), NAN>);
  // nan-not-constant-error@-1 {{non-type template argument is not a constant expression}}
  // nan-not-constant-note@-2 {{floating point arithmetic produces a NaN}}
  static_assert(is_same_val<CALL_BUILTIN(fmaxl, 123.456L, 789.012L), 789.012L>);

  // frexp
  static_assert(is_same_val<CALL_BUILTIN(frexp, 123.45, (int [1]){}), 123.45/128>);
  // cplusplus20andless-error@-1 {{non-type template argument is not a constant expression}}
  static_assert(is_same_val<CALL_BUILTIN(frexp, 0.0, (int [1]){}), 0.0>);
  // cplusplus20andless-error@-1 {{non-type template argument is not a constant expression}}
  static_assert(is_same_val<CALL_BUILTIN(frexp, -0.0, (int [1]){}), -0.0>);
  // cplusplus20andless-error@-1 {{non-type template argument is not a constant expression}}
  static_assert(is_same_val<CALL_BUILTIN(frexpf, NAN, (int [1]){}), NAN>);
  // nan-not-constant-error@-1 {{non-type template argument is not a constant expression}}
  // nan-not-constant-note@-2 {{floating point arithmetic produces a NaN}}
  // cplusplus20andless-error@-3 {{non-type template argument is not a constant expression}}
  static_assert(is_same_val<CALL_BUILTIN(frexpf, -NAN, (int [1]){}), -NAN>);
  // nan-not-constant-error@-1 {{non-type template argument is not a constant expression}}
  // nan-not-constant-note@-2 {{floating point arithmetic produces a NaN}}
  // cplusplus20andless-error@-3 {{non-type template argument is not a constant expression}}
  static_assert(is_same_val<CALL_BUILTIN(frexpf, INFINITY, (int [1]){}), INFINITY>);
  // cplusplus20andless-error@-1 {{non-type template argument is not a constant expression}}
  static_assert(is_same_val<CALL_BUILTIN(frexpf, -INFINITY, (int [1]){}), -INFINITY>);
  // cplusplus20andless-error@-1 {{non-type template argument is not a constant expression}}
  static_assert(is_same_val<CALL_BUILTIN(frexpl, 123.45L, (int [1]){}), 123.45L/128>);
  // cplusplus20andless-error@-1 {{non-type template argument is not a constant expression}}
  static_assert(is_same_val<CALL_BUILTIN(frexpl, 259.328L, (int [1]){}), 259.328L/512>);
  // cplusplus20andless-error@-1 {{non-type template argument is not a constant expression}}
  static_assert(is_same_val<CALL_BUILTIN(frexp, 3.5, (int [1]){}), 3.5/4>);
  // cplusplus20andless-error@-1 {{non-type template argument is not a constant expression}}

  return 0;
}
