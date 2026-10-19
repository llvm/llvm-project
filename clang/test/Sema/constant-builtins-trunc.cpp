// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s

// FIXME
auto y  = "// expected-no-diagnostics";

// trunc
// truncf
// truncl

//   ASSERT_NOT_CONSTEXPR_CXX23(std::trunc(1.0f) == 1.0f);
//   ASSERT_NOT_CONSTEXPR_CXX23(std::trunc(1.0) == 1.0);
//   ASSERT_NOT_CONSTEXPR_CXX23(std::trunc(1.0L) == 1.0L);
//   ASSERT_NOT_CONSTEXPR_CXX23(std::truncf(1.0f) == 1.0f);
//   ASSERT_NOT_CONSTEXPR_CXX23(std::truncl(1.0L) == 1.0L);


//   std::trunc(1.0f) 
//   std::trunc(1.0)
//   std::trunc(1.0L)
//   std::truncf(1.0f)
//   std::truncl(1.0L)


constexpr double NaN = __builtin_nan("");
constexpr double Inf = __builtin_inf();
constexpr double NegInf = -__builtin_inf();

static_assert(__builtin_trunc(+2.7f) == 2.0);
static_assert(__builtin_trunc(+2.7) == 2.0);
static_assert(__builtin_trunc(+2.7L) == 2.0L);
static_assert(__builtin_truncf(+2.7f) == 2.0f);
static_assert(__builtin_truncl(+2.7L) == 2.0L);
static_assert(__builtin_truncf16(+2.7f) == 2.0f);
static_assert(__builtin_truncf128(+2.7f) == 2.0f);

#define TRUNC_TEST_POSITIVE(T, FUNC) \
  static_assert(FUNC(T(2.7)) == T(2.0)); \
  static_assert(FUNC(T(2.0)) == T(2.0)); \
  static_assert(FUNC(T(0.0)) == T(0.0));

#define TRUNC_TEST_NEGATIVE(T, FUNC) \
  static_assert(FUNC(T(-2.7)) == T(-2.0)); \
  static_assert(FUNC(T(-2.0)) == T(-2.0)); \
  static_assert(FUNC(T(-0.0)) == T(-0.0));

#define TRUNC_TEST_SPECIAL(T, FUNC) \
  static_assert(FUNC(T(Inf)) == T(Inf)); \
  static_assert(FUNC(T(NegInf)) == T(NegInf)); \
  static_assert(__builtin_isnan(FUNC(NaN)));

#define LIST_TRUNC_TESTS(T, FUNC) \
    TRUNC_TEST_POSITIVE(T, FUNC) \
    TRUNC_TEST_NEGATIVE(T, FUNC) \
    TRUNC_TEST_SPECIAL(T, FUNC)


LIST_TRUNC_TESTS(float, __builtin_trunc);
LIST_TRUNC_TESTS(double, __builtin_trunc);
LIST_TRUNC_TESTS((long double), __builtin_trunc);
LIST_TRUNC_TESTS(__fp16, __builtin_truncf16)
#ifdef __FLOAT128__
LIST_TRUNC_TESTS(__float128, __builtin_truncf128)
#endif

static_assert(__builtin_isnan(__builtin_trunc(NaN)));
static_assert(__builtin_trunc(Inf) == Inf);
static_assert(__builtin_trunc(NegInf) == NegInf);
static_assert(__builtin_trunc(0.0) == 0.0);
static_assert(__builtin_trunc(-0.0) == -0.0);

// FIXME: remove, kept here temporarily
// expected-error@+1 {{static assertion failed}}
static_assert(0 == 1);
