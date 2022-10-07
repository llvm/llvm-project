// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// expected-no-diagnostics

constexpr double NaN = __builtin_nan("");
constexpr double Inf = __builtin_inf();
constexpr double NegInf = -__builtin_inf();

#define FMAX_TEST_SIMPLE(T, FUNC)                           \
    static_assert(T(6.7890) == FUNC(T(1.2345), T(6.7890))); \
    static_assert(T(6.7890) == FUNC(T(6.7890), T(1.2345)));

#define FMAX_TEST_NAN(T, FUNC)                          \
    static_assert(Inf == FUNC(NaN, Inf));               \
    static_assert(NegInf == FUNC(NegInf, NaN));         \
    static_assert(0.0 == FUNC(NaN, 0.0));               \
    static_assert(-0.0 == FUNC(-0.0, NaN));             \
    static_assert(T(-1.2345) == FUNC(NaN, T(-1.2345))); \
    static_assert(T(1.2345) == FUNC(T(1.2345), NaN));   \
    static_assert(__builtin_isnan(FUNC(NaN, NaN)));

#define FMAX_TEST_INF(T, FUNC)                  \
    static_assert(Inf == FUNC(NegInf, Inf));    \
    static_assert(Inf == FUNC(Inf, 0.0));       \
    static_assert(Inf == FUNC(-0.0, Inf));      \
    static_assert(Inf == FUNC(Inf, T(1.2345))); \
    static_assert(Inf == FUNC(T(-1.2345), Inf));

#define FMAX_TEST_NEG_INF(T, FUNC)                         \
    static_assert(Inf == FUNC(Inf, NegInf));               \
    static_assert(0.0 == FUNC(NegInf, 0.0));               \
    static_assert(-0.0 == FUNC(-0.0, NegInf));             \
    static_assert(T(-1.2345) == FUNC(NegInf, T(-1.2345))); \
    static_assert(T(1.2345) == FUNC(T(1.2345), NegInf));

#define FMAX_TEST_BOTH_ZERO(T, FUNC)       \
    static_assert(__builtin_copysign(1.0, FUNC(0.0, 0.0)) == 1.0);  \
    static_assert(__builtin_copysign(1.0, FUNC(-0.0, 0.0)) == 1.0);  \
    static_assert(__builtin_copysign(1.0, FUNC(0.0, -0.0)) == 1.0);  \
    static_assert(__builtin_copysign(1.0, FUNC(-0.0, -0.0)) == -1.0);

#define LIST_FMAX_TESTS(T, FUNC) \
    FMAX_TEST_SIMPLE(T, FUNC)    \
    FMAX_TEST_NAN(T, FUNC)       \
    FMAX_TEST_INF(T, FUNC)       \
    FMAX_TEST_NEG_INF(T, FUNC)   \
    FMAX_TEST_BOTH_ZERO(T, FUNC)

LIST_FMAX_TESTS(double, __builtin_fmax)
LIST_FMAX_TESTS(float, __builtin_fmaxf)
LIST_FMAX_TESTS((long double), __builtin_fmaxl)
LIST_FMAX_TESTS(__fp16, __builtin_fmaxf16)
#ifdef __FLOAT128__
LIST_FMAX_TESTS(__float128, __builtin_fmaxf128)
#endif
