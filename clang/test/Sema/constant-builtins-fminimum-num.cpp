// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s
// expected-no-diagnostics

constexpr double NaN = __builtin_nan("");
constexpr double SNaN = __builtin_nans("");
constexpr double Inf = __builtin_inf();
constexpr double NegInf = -__builtin_inf();

#define FMINIMUMNUM_TEST_SIMPLE(T, FUNC)                           \
    static_assert(T(1.2345) == FUNC(T(1.2345), T(6.7890))); \
    static_assert(T(1.2345) == FUNC(T(6.7890), T(1.2345)));

#define FMINIMUMNUM_TEST_NAN(T, FUNC)                          \
    static_assert(Inf == FUNC(NaN, Inf));               \
    static_assert(NegInf == FUNC(NegInf, NaN));         \
    static_assert(0.0 == FUNC(NaN, 0.0));               \
    static_assert(-0.0 == FUNC(-0.0, NaN));             \
    static_assert(T(-1.2345) == FUNC(NaN, T(-1.2345))); \
    static_assert(T(1.2345) == FUNC(T(1.2345), NaN));   \
    static_assert(__builtin_isnan(FUNC(NaN, NaN)));

#define FMINIMUMNUM_TEST_SNAN(T, FUNC)                          \
    static_assert(Inf == FUNC(SNaN, Inf));               \
    static_assert(NegInf == FUNC(NegInf, SNaN));         \
    static_assert(0.0 == FUNC(SNaN, 0.0));               \
    static_assert(-0.0 == FUNC(-0.0, SNaN));             \
    static_assert(T(-1.2345) == FUNC(SNaN, T(-1.2345))); \
    static_assert(T(1.2345) == FUNC(T(1.2345), SNaN));   \
    static_assert(__builtin_isnan(FUNC(SNaN, SNaN)));    \
    static_assert(__builtin_isnan(FUNC(NaN, SNaN)));    \
    static_assert(!__builtin_issignaling(FUNC(SNaN, SNaN)));  \
    static_assert(!__builtin_issignaling(FUNC(NaN, SNaN)));

#define FMINIMUMNUM_TEST_INF(T, FUNC)                        \
    static_assert(NegInf == FUNC(NegInf, Inf));       \
    static_assert(0.0 == FUNC(Inf, 0.0));             \
    static_assert(-0.0 == FUNC(-0.0, Inf));           \
    static_assert(T(1.2345) == FUNC(Inf, T(1.2345))); \
    static_assert(T(-1.2345) == FUNC(T(-1.2345), Inf));

#define FMINIMUMNUM_TEST_NEG_INF(T, FUNC)                     \
    static_assert(NegInf == FUNC(Inf, NegInf));        \
    static_assert(NegInf == FUNC(NegInf, 0.0));        \
    static_assert(NegInf == FUNC(-0.0, NegInf));       \
    static_assert(NegInf == FUNC(NegInf, T(-1.2345))); \
    static_assert(NegInf == FUNC(T(1.2345), NegInf));

#define FMINIMUMNUM_TEST_BOTH_ZERO(T, FUNC)                                 \
    static_assert(__builtin_copysign(1.0, FUNC(0.0, 0.0)) == 1.0);   \
    static_assert(__builtin_copysign(1.0, FUNC(-0.0, 0.0)) == -1.0); \
    static_assert(__builtin_copysign(1.0, FUNC(0.0, -0.0)) == -1.0); \
    static_assert(__builtin_copysign(1.0, FUNC(-0.0, -0.0)) == -1.0);

#define LIST_FMINIMUMNUM_TESTS(T, FUNC) \
    FMINIMUMNUM_TEST_SIMPLE(T, FUNC)    \
    FMINIMUMNUM_TEST_NAN(T, FUNC)       \
    FMINIMUMNUM_TEST_SNAN(T, FUNC)      \
    FMINIMUMNUM_TEST_INF(T, FUNC)       \
    FMINIMUMNUM_TEST_NEG_INF(T, FUNC)   \
    FMINIMUMNUM_TEST_BOTH_ZERO(T, FUNC)

LIST_FMINIMUMNUM_TESTS(double, __builtin_fminimum_num)
LIST_FMINIMUMNUM_TESTS(float, __builtin_fminimum_numf)
LIST_FMINIMUMNUM_TESTS((long double), __builtin_fminimum_numl)
LIST_FMINIMUMNUM_TESTS(__fp16, __builtin_fminimum_numf16)
#ifdef __FLOAT128__
LIST_FMINIMUMNUM_TESTS(__float128, __builtin_fminimum_numf128)
#endif
