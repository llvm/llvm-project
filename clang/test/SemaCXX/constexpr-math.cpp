// RUN: %clang_cc1 -DWIN -verify -std=c++23 -fsyntax-only  %s
// RUN: %clang_cc1 -verify -std=c++23 -fsyntax-only  %s

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

int main() {
  int i;

  // fmin
  static_assert(is_same_val<__builtin_fmin(15.24, 1.3), 1.3>);
  static_assert(is_same_val<__builtin_fmin(-0.0, +0.0), -0.0>);
  static_assert(is_same_val<__builtin_fmin(+0.0, -0.0), -0.0>);
  static_assert(is_same_val<__builtin_fminf(NAN, -1), -1.f>);
  static_assert(is_same_val<__builtin_fminf(+INFINITY, 0), 0.f>);
  static_assert(is_same_val<__builtin_fminf(-INFINITY, 0), -INFINITY>);
  static_assert(is_same_val<__builtin_fminf(NAN, NAN), NAN>);
  static_assert(is_same_val<__builtin_fminl(123.456L, 789.012L), 123.456L>);

  // fmax
  static_assert(is_same_val<__builtin_fmax(15.24, 1.3), 15.24>);
  static_assert(is_same_val<__builtin_fmax(-0.0, +0.0), +0.0>);
  static_assert(is_same_val<__builtin_fmax(+0.0, -0.0), +0.0>);
  static_assert(is_same_val<__builtin_fmaxf(NAN, -1), -1.f>);
  static_assert(is_same_val<__builtin_fmaxf(+INFINITY, 0), INFINITY>);
  static_assert(is_same_val<__builtin_fmaxf(-INFINITY, 0), 0.f>);
  static_assert(is_same_val<__builtin_fmaxf(NAN, NAN), NAN>);
  static_assert(is_same_val<__builtin_fmaxl(123.456L, 789.012L), 789.012L>);

  // frexp
  static_assert(is_same_val<__builtin_frexp(123.45, &i), 123.45/128>);
  static_assert(is_same_val<__builtin_frexp(0.0, &i), 0.0>);
  static_assert(is_same_val<__builtin_frexp(-0.0, &i), -0.0>);
  static_assert(is_same_val<__builtin_frexpf(NAN, &i), NAN>);
  static_assert(is_same_val<__builtin_frexpf(-NAN, &i), -NAN>);
  static_assert(is_same_val<__builtin_frexpf(INFINITY, &i), INFINITY>);
  static_assert(is_same_val<__builtin_frexpf(-INFINITY, &i), -INFINITY>);
  static_assert(is_same_val<__builtin_frexpl(123.45L, &i), 123.45L/128>);

  return 0;
}
