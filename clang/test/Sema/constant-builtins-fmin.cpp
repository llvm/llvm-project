// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// expected-no-diagnostics

constexpr double NaN = __builtin_nan("");
constexpr double Inf = __builtin_inf();
constexpr double NegInf = -__builtin_inf();

#define FMIN_TEST_SIMPLE(T, FUNC)                           \
    static_assert(T(1.2345) == FUNC(T(1.2345), T(6.7890))); \
    static_assert(T(1.2345) == FUNC(T(6.7890), T(1.2345)));

#define FMIN_TEST_NAN(T, FUNC)                          \
    static_assert(Inf == FUNC(NaN, Inf));               \
    static_assert(NegInf == FUNC(NegInf, NaN));         \
    static_assert(0.0 == FUNC(NaN, 0.0));               \
    static_assert(-0.0 == FUNC(-0.0, NaN));             \
    static_assert(T(-1.2345) == FUNC(NaN, T(-1.2345))); \
    static_assert(T(1.2345) == FUNC(T(1.2345), NaN));   \
    static_assert(__builtin_isnan(FUNC(NaN, NaN)));

#define FMIN_TEST_INF(T, FUNC)                        \
    static_assert(NegInf == FUNC(NegInf, Inf));       \
    static_assert(0.0 == FUNC(Inf, 0.0));             \
    static_assert(-0.0 == FUNC(-0.0, Inf));           \
    static_assert(T(1.2345) == FUNC(Inf, T(1.2345))); \
    static_assert(T(-1.2345) == FUNC(T(-1.2345), Inf));

#define FMIN_TEST_NEG_INF(T, FUNC)                     \
    static_assert(NegInf == FUNC(Inf, NegInf));        \
    static_assert(NegInf == FUNC(NegInf, 0.0));        \
    static_assert(NegInf == FUNC(-0.0, NegInf));       \
    static_assert(NegInf == FUNC(NegInf, T(-1.2345))); \
    static_assert(NegInf == FUNC(T(1.2345), NegInf));

#define FMIN_TEST_BOTH_ZERO(T, FUNC)                                 \
    static_assert(__builtin_copysign(1.0, FUNC(0.0, 0.0)) == 1.0);   \
    static_assert(__builtin_copysign(1.0, FUNC(-0.0, 0.0)) == -1.0); \
    static_assert(__builtin_copysign(1.0, FUNC(0.0, -0.0)) == -1.0); \
    static_assert(__builtin_copysign(1.0, FUNC(-0.0, -0.0)) == -1.0);

#define LIST_FMIN_TESTS(T, FUNC) \
    FMIN_TEST_SIMPLE(T, FUNC)    \
    FMIN_TEST_NAN(T, FUNC)       \
    FMIN_TEST_INF(T, FUNC)       \
    FMIN_TEST_NEG_INF(T, FUNC)   \
    FMIN_TEST_BOTH_ZERO(T, FUNC)

LIST_FMIN_TESTS(double, __builtin_fmin)
LIST_FMIN_TESTS(float, __builtin_fminf)
LIST_FMIN_TESTS((long double), __builtin_fminl)
LIST_FMIN_TESTS(__fp16, __builtin_fminf16)
#ifdef __FLOAT128__
LIST_FMIN_TESTS(__float128, __builtin_fminf128)
#endif
