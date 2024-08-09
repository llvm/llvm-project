// RUN: %clang_analyze_cc1 -triple x86_64-unknown-unknown -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection

#define __UINT_MAX__ (__INT_MAX__  * 2U + 1U)

void clang_analyzer_dump_int(int);
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

void test_add_overflow_contraints(int a, int b)
{
   int res;

   if (a != 10)
     return;
   if (b != 0)
     return;

   if (__builtin_add_overflow(a, b, &res)) {
     clang_analyzer_warnIfReached();
     return;
   }

   clang_analyzer_dump_int(res); //expected-warning{{10 S32b}}
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

   if (a > 10)
     return;
   if (b > 10)
     return;

   // clang_analyzer_eval(a + b < 30); <--- Prints 1 and 0, but why ???

   if (__builtin_uadd_overflow(a, b, &res)) {
     /* clang_analyzer_warnIfReached(); */
     return;
   }
}

// TODO: more tests after figuring out what's going on.
