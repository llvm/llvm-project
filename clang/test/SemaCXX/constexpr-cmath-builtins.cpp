// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,new -std=c++20 %s
// RUN: %clang_cc1 -verify=expected -std=c++20 %s

constexpr bool isNegativeZero(double x) {
  return x == 0.0 && __builtin_copysign(1.0, x) < 0.0;
}
constexpr bool isPositiveZero(double x) {
  return x == 0.0 && __builtin_copysign(1.0, x) > 0.0;
}

// round tests
static_assert(__builtin_round(1.1) == 1.0);
static_assert(__builtin_round(1.5) == 2.0);
static_assert(__builtin_round(1.9) == 2.0);
static_assert(__builtin_round(-1.5) == -2.0);
static_assert(__builtin_roundf16(1.5f16) == 2.0f16);
static_assert(__builtin_roundf128(1.5) == 2.0);
static_assert(__builtin_isnan(__builtin_round(__builtin_nan(""))));
static_assert(__builtin_isinf(__builtin_round(__builtin_inf())));

// lround tests
static_assert(__builtin_lround(1.1) == 1);
static_assert(__builtin_lround(1.5) == 2);
static_assert(__builtin_lround(-1.5) == -2);
static_assert(__builtin_lroundf128(1.5) == 2);

// llround tests
static_assert(__builtin_llround(1.1) == 1LL);
static_assert(__builtin_llround(1.5) == 2LL);
static_assert(__builtin_llround(-1.5) == -2LL);
static_assert(__builtin_llroundf128(1.5) == 2LL);

// ceil tests
static_assert(__builtin_ceil(1.1) == 2.0);
static_assert(__builtin_ceil(-1.1) == -1.0);
static_assert(__builtin_ceilf(1.1f) == 2.0f);
static_assert(__builtin_ceilf16(1.1f16) == 2.0f16);
static_assert(__builtin_ceilf128(1.1) == 2.0);
static_assert(__builtin_isnan(__builtin_ceil(__builtin_nan(""))));
static_assert(__builtin_isinf(__builtin_ceil(__builtin_inf())));

// floor tests
static_assert(__builtin_floor(1.1) == 1.0);
static_assert(__builtin_floor(-1.1) == -2.0);
static_assert(__builtin_floorf(1.1f) == 1.0f);
static_assert(__builtin_floorf16(1.1f16) == 1.0f16);
static_assert(__builtin_floorf128(1.1) == 1.0);
static_assert(__builtin_isnan(__builtin_floor(__builtin_nan(""))));
static_assert(__builtin_isinf(__builtin_floor(__builtin_inf())));

// trunc tests
static_assert(__builtin_trunc(1.1) == 1.0);
static_assert(__builtin_trunc(-1.1) == -1.0);
static_assert(__builtin_truncf(1.1f) == 1.0f);
static_assert(__builtin_truncf16(1.1f16) == 1.0f16);
static_assert(__builtin_truncf128(1.1) == 1.0);
static_assert(__builtin_isnan(__builtin_trunc(__builtin_nan(""))));
static_assert(__builtin_isinf(__builtin_trunc(__builtin_inf())));

// roundeven tests
static_assert(__builtin_roundeven(1.5) == 2.0);
static_assert(__builtin_roundeven(2.5) == 2.0);
static_assert(__builtin_roundeven(-1.5) == -2.0);
static_assert(__builtin_roundeven(-2.5) == -2.0);
static_assert(__builtin_roundevenf(1.5f) == 2.0f);
static_assert(__builtin_roundevenl(1.5l) == 2.0l);
static_assert(__builtin_roundevenf128(1.5) == 2.0);
static_assert(__builtin_roundeven(0.5) == 0.0);
static_assert(isNegativeZero(__builtin_roundeven(-0.5)));
static_assert(__builtin_isnan(__builtin_roundeven(__builtin_nan(""))));
static_assert(__builtin_isinf(__builtin_roundeven(__builtin_inf())));

// fdim tests
static_assert(__builtin_fdim(3.0, 1.0) == 2.0);
static_assert(__builtin_fdim(1.0, 3.0) == 0.0);
static_assert(__builtin_fdimf(3.0f, 1.0f) == 2.0f);
static_assert(__builtin_fdimf128(3.0, 1.0) == 2.0);
static_assert(__builtin_isnan(__builtin_fdim(__builtin_nan(""), 1.0)));
static_assert(__builtin_isnan(__builtin_fdim(1.0, __builtin_nan(""))));
static_assert(__builtin_isinf(__builtin_fdim(__builtin_inf(), 0.0)));
static_assert(__builtin_fdim(__builtin_inf(), __builtin_inf()) == 0.0);

// fma tests
static_assert(__builtin_fma(2.0, 3.0, 4.0) == 10.0);
static_assert(__builtin_fmaf(2.0f, 3.0f, 4.0f) == 10.0f);
static_assert(__builtin_fmaf16(2.0f16, 3.0f16, 4.0f16) == 10.0f16);
static_assert(__builtin_fmaf128(2.0, 3.0, 4.0) == 10.0);
static_assert(__builtin_isnan(__builtin_fma(__builtin_nan(""), 2.0, 3.0)));
static_assert(__builtin_isnan(__builtin_fma(1.0, __builtin_nan(""), 3.0)));
static_assert(__builtin_isnan(__builtin_fma(1.0, 2.0, __builtin_nan(""))));
static_assert(__builtin_isinf(__builtin_fma(__builtin_inf(), 2.0, 3.0)));

// fmod tests
static_assert(__builtin_fmod(5.5, 3.0) == 2.5);
static_assert(__builtin_fmodf(5.5f, 3.0f) == 2.5f);
static_assert(__builtin_fmodf16(5.5f16, 3.0f16) == 2.5f16);
static_assert(__builtin_fmodf128(5.5, 3.0) == 2.5);
static_assert(__builtin_isnan(__builtin_fmod(__builtin_nan(""), 2.0)));
static_assert(__builtin_isnan(__builtin_fmod(2.0, __builtin_nan(""))));

// remainder tests
static_assert(__builtin_remainder(5.5, 3.0) == -0.5);
static_assert(__builtin_remainderf(5.5f, 3.0f) == -0.5f);
static_assert(__builtin_remainderf128(5.5, 3.0) == -0.5);
static_assert(__builtin_isnan(__builtin_remainder(__builtin_nan(""), 2.0)));
static_assert(__builtin_isnan(__builtin_remainder(2.0, __builtin_nan(""))));

// nextafter tests
static_assert(__builtin_nextafter(1.0, 2.0) > 1.0);
static_assert(__builtin_nextafter(1.0, 0.0) < 1.0);
static_assert(__builtin_nextafter(1.0, 1.0) == 1.0);
static_assert(__builtin_nextafter(0.0, 1.0) > 0.0); // new-error {{static assertion expression is not an integral constant expression}} \
                                                     // new-note {{floating point arithmetic produces an infinity}}
static_assert(__builtin_nextafter(0.0, -1.0) < 0.0); // new-error {{static assertion expression is not an integral constant expression}} \
                                                     // new-note {{floating point arithmetic produces an infinity}}
static_assert(isPositiveZero(__builtin_nextafter(-0.0, 0.0)));
static_assert(__builtin_nextafterf128(1.0, 2.0) > 1.0);
static_assert(__builtin_isnan(__builtin_nextafter(__builtin_nan(""), 2.0)));
static_assert(__builtin_isnan(__builtin_nextafter(2.0, __builtin_nan(""))));
static_assert(__builtin_isinf(__builtin_nextafter(__builtin_inf(), __builtin_inf())));

// nexttoward tests
static_assert(__builtin_nexttoward(1.0, 2.0L) > 1.0);
static_assert(__builtin_nexttoward(1.0, 1.0L) == 1.0);
static_assert(__builtin_nexttowardf128(1.0, 2.0L) > 1.0);
static_assert(__builtin_isnan(__builtin_nexttoward(__builtin_nan(""), 2.0L)));
static_assert(__builtin_isnan(__builtin_nexttoward(2.0, __builtin_nan(""))));

// scalbn tests
static_assert(__builtin_scalbn(1.0, 2) == 4.0);
static_assert(__builtin_scalbnf(1.0f, -1) == 0.5f);
static_assert(__builtin_scalbnf128(1.0, 2) == 4.0);
static_assert(__builtin_scalbn(0.0, 2) == 0.0);
static_assert(__builtin_scalbn(1.0, 0) == 1.0);
static_assert(__builtin_isnan(__builtin_scalbn(__builtin_nan(""), 2)));
static_assert(__builtin_isinf(__builtin_scalbn(__builtin_inf(), 2)));

// scalbln tests
static_assert(__builtin_scalbln(1.0, 2L) == 4.0);
static_assert(__builtin_scalblnf128(1.0, 2L) == 4.0);
static_assert(__builtin_isnan(__builtin_scalbln(__builtin_nan(""), 2L)));
static_assert(__builtin_isinf(__builtin_scalbln(__builtin_inf(), 2L)));

// ldexp tests
static_assert(__builtin_ldexp(1.0, 3) == 8.0);
static_assert(__builtin_ldexpf16(1.0f16, 3) == 8.0f16);
static_assert(__builtin_ldexpf128(1.0, 3) == 8.0);
static_assert(__builtin_isnan(__builtin_ldexp(__builtin_nan(""), 2)));
static_assert(__builtin_isinf(__builtin_ldexp(__builtin_inf(), 2)));

// ilogb tests
static_assert(__builtin_ilogb(1.0) == 0);
static_assert(__builtin_ilogb(2.0) == 1);
static_assert(__builtin_ilogb(0.5) == -1);
static_assert(__builtin_ilogbf(8.0f) == 3);
static_assert(__builtin_ilogbf128(8.0) == 3);
static_assert(__builtin_ilogb(0.) == (-__INT_MAX__));
static_assert(__builtin_ilogb(__builtin_nan("")) == (-__INT_MAX__ - 1));
static_assert(__builtin_ilogb(__builtin_inf()) == __INT_MAX__);

// remquo tests
constexpr double test_remquo(double x, double y) {
  int quo = 0;
  double rem = __builtin_remquo(x, y, &quo);
  return rem;
}
static_assert(test_remquo(10.0, 3.0) == 1.0);

constexpr int test_remquo_quo(double x, double y) {
  int quo = 0;
  __builtin_remquo(x, y, &quo);
  return quo;
}
static_assert(test_remquo_quo(10.0, 3.0) == 3);
static_assert(test_remquo_quo(10.0, -3.0) == -3);

// remquo NaN cases (per C standard / cppreference):
// - x or y is NaN
// - x is +/-inf
// - y is +/-0
static_assert(__builtin_isnan(test_remquo(__builtin_nan(""), 2.0)));
static_assert(__builtin_isnan(test_remquo(2.0, __builtin_nan(""))));
static_assert(__builtin_isnan(test_remquo(__builtin_nan(""), __builtin_nan(""))));

// frexp tests
constexpr double test_frexp_val(double x) {
  int exp;
  return __builtin_frexp(x, &exp);
}
static_assert(test_frexp_val(8.0) == 0.5);

constexpr int test_frexp_exp(double x) {
  int exp;
  __builtin_frexp(x, &exp);
  return exp;
}
static_assert(test_frexp_exp(8.0) == 4);

// frexp special cases: +/- 0
static_assert(test_frexp_val(0.0) == 0.0);
static_assert(isNegativeZero(test_frexp_val(-0.0)));
static_assert(test_frexp_exp(0.0) == 0);
static_assert(test_frexp_exp(-0.0) == 0);
// NaN and Inf: LLVM does not specify the exponent value for these cases.
static_assert(__builtin_isnan(test_frexp_val(__builtin_nan(""))));
static_assert(__builtin_isinf(test_frexp_val(__builtin_inf())));
constexpr int frexp_nan_exp = test_frexp_exp(__builtin_nan(""));
constexpr int frexp_inf_exp = test_frexp_exp(__builtin_inf());
constexpr int frexp_neg_inf_exp = test_frexp_exp(-__builtin_inf());
static_assert(test_frexp_val(0.5) == 0.5);
static_assert(test_frexp_exp(0.5) == 0);
static_assert(test_frexp_val(1.0) == 0.5);
static_assert(test_frexp_exp(1.0) == 1);

// modf tests
constexpr double test_modf_val(double x) {
  double iptr;
  return __builtin_modf(x, &iptr);
}
static_assert(test_modf_val(1.5) == 0.5);
static_assert(test_modf_val(-1.5) == -0.5);
static_assert(isNegativeZero(test_modf_val(-0.0)));

constexpr double test_modf_iptr(double x) {
  double iptr;
  __builtin_modf(x, &iptr);
  return iptr;
}
static_assert(test_modf_iptr(1.5) == 1.0);
static_assert(test_modf_iptr(-1.5) == -1.0);

// modf special values
static_assert(test_modf_val(__builtin_inf()) == 0.0);
static_assert(__builtin_isinf(test_modf_iptr(__builtin_inf())));


namespace LRoundDiagnostic {
  constexpr int i = __builtin_lround(1e30); // expected-error {{constexpr variable 'i' must be initialized by a constant expression}} \
                                            // expected-note {{floating point arithmetic produces an infinity}}
}

// NaN payload preservation tests.
template <typename T> struct Bytes {
  unsigned char bytes[sizeof(T)];
  bool operator==(const Bytes &) const = default;
};
template <typename T> constexpr bool bytesEqual(T a, T b) {
  return __builtin_bit_cast(Bytes<T>, a) == __builtin_bit_cast(Bytes<T>, b);
}

constexpr double nanWithNonZeroPayload = __builtin_nan("0x1337");

// fdim NaN payload preservation
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_fdim(nanWithNonZeroPayload, 0.)));
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_fdim(0., nanWithNonZeroPayload)));

// round NaN payload preservation
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_round(nanWithNonZeroPayload)));

// ceil NaN payload preservation
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_ceil(nanWithNonZeroPayload)));

// floor NaN payload preservation
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_floor(nanWithNonZeroPayload)));

// trunc NaN payload preservation
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_trunc(nanWithNonZeroPayload)));

// roundeven NaN payload preservation
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_roundeven(nanWithNonZeroPayload)));

// fmod NaN payload preservation
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_fmod(nanWithNonZeroPayload, 2.0)));
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_fmod(2.0, nanWithNonZeroPayload)));

// remainder NaN payload preservation
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_remainder(nanWithNonZeroPayload, 2.0)));
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_remainder(2.0, nanWithNonZeroPayload)));

// scalbn NaN payload preservation
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_scalbn(nanWithNonZeroPayload, 2)));

// ldexp NaN payload preservation
static_assert(bytesEqual(nanWithNonZeroPayload,
                         __builtin_ldexp(nanWithNonZeroPayload, 2)));

// modf NaN payload preservation
constexpr double test_modf_nan() {
  double iptr;
  return __builtin_modf(nanWithNonZeroPayload, &iptr);
}
static_assert(bytesEqual(nanWithNonZeroPayload, test_modf_nan()));

// frexp NaN payload preservation
constexpr double test_frexp_nan() {
  int exp;
  return __builtin_frexp(nanWithNonZeroPayload, &exp);
}
static_assert(bytesEqual(nanWithNonZeroPayload, test_frexp_nan()));
