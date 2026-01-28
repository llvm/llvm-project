// RUN: %clang_analyze_cc1 -triple x86_64-unknown-unknown -verify %s -Wno-strict-prototypes \
// RUN:   -analyzer-checker=core,debug.ExprInspection,alpha.core.BoolAssignment

#define __UINT_MAX__ (__INT_MAX__ * 2U + 1U)
#define __INT_MIN__  (-__INT_MAX__ - 1)
#define __UINT128_MAX__ ((__uint128_t)((__int128_t)(-1L)))
#define __INT128_MAX__ ((__int128_t)(__UINT128_MAX__ >> 1))
#define __UBITINT_MAX__(BITS) ((unsigned _BitInt(BITS))-1)
#define __BITINT_MAX__(BITS) ((_BitInt(BITS))(__UBITINT_MAX__(BITS) >> 1))

void clang_analyzer_dump(/*not specified*/);
void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached(void);

void test_add_nooverflow(void)
{
   int res;

   if (__builtin_add_overflow(10, 20, &res)) {
     clang_analyzer_warnIfReached(); // no-wrapping happened
     return;
   }

   clang_analyzer_dump(res); //expected-warning{{30 S32b}}
}

void test_add_overflow(void)
{
   int res;

   if (__builtin_add_overflow(__INT_MAX__, 1, &res)) {
     clang_analyzer_dump(res); //expected-warning{{-2147483648 S32b}}
     return;
   }

   clang_analyzer_warnIfReached();
}

void test_add_underoverflow(void)
{
   int res;

   if (__builtin_add_overflow(__INT_MIN__, -1, &res)) {
     clang_analyzer_dump(res); //expected-warning{{2147483647 S32b}}
     return;
   }

   clang_analyzer_warnIfReached();
}

void test_sub_underflow(void)
{
   int res;

   if (__builtin_sub_overflow(__INT_MIN__, 10, &res)) {
     return;
   }

   clang_analyzer_warnIfReached();
}

void test_sub_overflow(void)
{
   int res;

   if (__builtin_sub_overflow(__INT_MAX__, -1, &res)) {
     return;
   }

   clang_analyzer_warnIfReached();
}

void test_sub_nooverflow(void)
{
   int res;

   if (__builtin_sub_overflow(__INT_MAX__, 1, &res)) {
     clang_analyzer_warnIfReached();
     return;
   }

   clang_analyzer_dump(res); //expected-warning{{2147483646 S32b}}
}

void test_mul_overflow(void)
{
   int res;

   if (__builtin_mul_overflow(__INT_MAX__, 2, &res)) {
     return;
   }

   clang_analyzer_warnIfReached();
}

void test_mul_underflow(void)
{
   int res;

   if (__builtin_mul_overflow(__INT_MIN__, -2, &res)) {
     return;
   }

   clang_analyzer_warnIfReached();
}

void test_mul_nooverflow(void)
{
   int res;

   if (__builtin_mul_overflow(10, -2, &res)) {
     clang_analyzer_warnIfReached();
     return;
   }

   clang_analyzer_dump(res); //expected-warning{{-20 S32b}}
}

void test_nooverflow_diff_types(void)
{
   long res;

   // This is not an overflow, since result type is long.
   if (__builtin_add_overflow(__INT_MAX__, 1, &res)) {
     clang_analyzer_warnIfReached();
     return;
   }

   clang_analyzer_dump(res); //expected-warning{{2147483648 S64b}}
}

void test_uaddll_overflow_contraints(unsigned long a, unsigned long b)
{
   unsigned long long res;

   if (a != 10)
     return;
   if (b != 10)
     return;

   if (__builtin_uaddll_overflow(a, b, &res)) {
     clang_analyzer_warnIfReached();
     return;
   }
}

void test_uadd_overflow_contraints(unsigned a, unsigned b)
{
   unsigned res;

   if (a > 5)
     return;
   if (b != 10)
     return;

   if (__builtin_uadd_overflow(a, b, &res)) {
     clang_analyzer_warnIfReached();
     return;
   }
}

void test_bool_assign(void)
{
    int res;

    // Reproduce issue from GH#111147. __builtin_*_overflow functions
    // should return _Bool, but not int.
    _Bool ret = __builtin_mul_overflow(10, 20, &res); // no crash
}

void no_crash_with_int128(__int128_t a, __int128_t b) {
  __int128_t result = 0;
  (void)__builtin_add_overflow(a, b, &result); // no-crash
}

void no_crash_with_uint128(__uint128_t a, __uint128_t b) {
  __uint128_t result = 0;
  (void)__builtin_add_overflow(a, b, &result); // no-crash
}

void no_crash_with_bigint(_BitInt(111) a, _BitInt(111) b) {
  _BitInt(111) result = 0;
  (void)__builtin_add_overflow(a, b, &result); // no-crash
}

void test_add_overflow_128(void) {
  __int128_t res;

  if (__builtin_add_overflow(__INT128_MAX__, 1, &res)) {
    clang_analyzer_dump(res); // expected-warning {{-170141183460469231731687303715884105728 S128b}}
    return;
  }

  clang_analyzer_warnIfReached(); // no-warning: we always get an overflow, thus choose the other branch
}

void test_add_overflow_u111(void) {
  unsigned _BitInt(111) res;

  if (__builtin_add_overflow(__UBITINT_MAX__(111), 1, &res)) {
    clang_analyzer_dump(res); // expected-warning {{0 U111b}}
    return;
  }

  clang_analyzer_warnIfReached(); // no-warning: we always get an overflow, thus choose the other branch
}

void test_add_overflow_s111(void) {
  _BitInt(111) res;

  if (__builtin_add_overflow(__BITINT_MAX__(111), 1, &res)) {
    clang_analyzer_dump(res); // expected-warning {{-1298074214633706907132624082305024 S111b}}
    return;
  }

  clang_analyzer_warnIfReached(); // no-warning: we always get an overflow, thus choose the other branch
}
