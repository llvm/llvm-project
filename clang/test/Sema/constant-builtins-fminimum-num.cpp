// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// FIXME: %clang_cc1 -std=c++17 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s
// expected-no-diagnostics

constexpr double NaN = __builtin_nan("");
constexpr double Inf = __builtin_inf();
constexpr double NegInf = -__builtin_inf();

#define FMAXIMUMNUM_TEST_SIMPLE(T, FUNC)                           \
    static_assert(T(1.2345) == FUNC(T(1.2345), T(6.7890))); \
    static_assert(T(1.2345) == FUNC(T(6.7890), T(1.2345)));

#define FMAXIMUMNUM_TEST_NAN(T, FUNC)                          \
    static_assert(Inf == FUNC(NaN, Inf));               \
    static_assert(NegInf == FUNC(NegInf, NaN));         \
    static_assert(0.0 == FUNC(NaN, 0.0));               \
    static_assert(-0.0 == FUNC(-0.0, NaN));             \
    static_assert(T(-1.2345) == FUNC(NaN, T(-1.2345))); \
    static_assert(T(1.2345) == FUNC(T(1.2345), NaN));   \
    static_assert(__builtin_isnan(FUNC(NaN, NaN)));

#define FMAXIMUMNUM_TEST_INF(T, FUNC)                        \
    static_assert(NegInf == FUNC(NegInf, Inf));       \
    static_assert(0.0 == FUNC(Inf, 0.0));             \
    static_assert(-0.0 == FUNC(-0.0, Inf));           \
    static_assert(T(1.2345) == FUNC(Inf, T(1.2345))); \
    static_assert(T(-1.2345) == FUNC(T(-1.2345), Inf));

#define FMAXIMUMNUM_TEST_NEG_INF(T, FUNC)                     \
    static_assert(NegInf == FUNC(Inf, NegInf));        \
    static_assert(NegInf == FUNC(NegInf, 0.0));        \
    static_assert(NegInf == FUNC(-0.0, NegInf));       \
    static_assert(NegInf == FUNC(NegInf, T(-1.2345))); \
    static_assert(NegInf == FUNC(T(1.2345), NegInf));

#define FMAXIMUMNUM_TEST_BOTH_ZERO(T, FUNC)                                 \
    static_assert(__builtin_copysign(1.0, FUNC(0.0, 0.0)) == 1.0);   \
    static_assert(__builtin_copysign(1.0, FUNC(-0.0, 0.0)) == -1.0); \
    static_assert(__builtin_copysign(1.0, FUNC(0.0, -0.0)) == -1.0); \
    static_assert(__builtin_copysign(1.0, FUNC(-0.0, -0.0)) == -1.0);

#define LIST_FMAXIMUMNUM_TESTS(T, FUNC) \
    FMAXIMUMNUM_TEST_SIMPLE(T, FUNC)    \
    FMAXIMUMNUM_TEST_NAN(T, FUNC)       \
    FMAXIMUMNUM_TEST_INF(T, FUNC)       \
    FMAXIMUMNUM_TEST_NEG_INF(T, FUNC)   \
    FMAXIMUMNUM_TEST_BOTH_ZERO(T, FUNC)

LIST_FMAXIMUMNUM_TESTS(double, __builtin_fminimum_num)
LIST_FMAXIMUMNUM_TESTS(float, __builtin_fminimum_numf)
LIST_FMAXIMUMNUM_TESTS((long double), __builtin_fminimum_numl)
LIST_FMAXIMUMNUM_TESTS(__fp16, __builtin_fminimum_numf16)
#ifdef __FLOAT128__
LIST_FMAXIMUMNUM_TESTS(__float128, __builtin_fminimum_numf128)
#endif
