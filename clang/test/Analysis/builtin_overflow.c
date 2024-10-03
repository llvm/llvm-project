// RUN: %clang_analyze_cc1 -triple x86_64-unknown-unknown -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection

#define __UINT_MAX__ (__INT_MAX__ * 2U + 1U)
#define __INT_MIN__  (-__INT_MAX__ - 1)

void clang_analyzer_dump_int(int);
void clang_analyzer_dump_long(long);
void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached(void);

void test_add_nooverflow(void)
{
   int res;

   if (__builtin_add_overflow(10, 20, &res)) {
     clang_analyzer_warnIfReached();
     return;
   }

   clang_analyzer_dump_int(res); //expected-warning{{30 S32b}}
}

void test_add_overflow(void)
{
   int res;

   if (__builtin_add_overflow(__INT_MAX__, 1, &res)) {
     clang_analyzer_dump_int(res); //expected-warning{{1st function call argument is an uninitialized value}}
     return;
   }

   clang_analyzer_warnIfReached();
}

void test_add_underoverflow(void)
{
   int res;

   if (__builtin_add_overflow(__INT_MIN__, -1, &res)) {
     clang_analyzer_dump_int(res); //expected-warning{{1st function call argument is an uninitialized value}}
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

   clang_analyzer_dump_int(res); //expected-warning{{2147483646 S32b}}
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

   clang_analyzer_dump_int(res); //expected-warning{{-20 S32b}}
}

void test_nooverflow_diff_types(void)
{
   long res;

   // This is not an overflow, since result type is long.
   if (__builtin_add_overflow(__INT_MAX__, 1, &res)) {
     clang_analyzer_warnIfReached();
     return;
   }

   clang_analyzer_dump_long(res); //expected-warning{{2147483648 S64b}}
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
