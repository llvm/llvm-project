// RUN: %clang_analyze_cc1 -triple x86_64-unknown-unknown -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection,alpha.core.BoolAssignment

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
     clang_analyzer_dump_int(res); //expected-warning{{-2147483648 S32b}}
     return;
   }

   clang_analyzer_warnIfReached();
}

void test_add_underoverflow(void)
{
   int res;

   if (__builtin_add_overflow(__INT_MIN__, -1, &res)) {
     clang_analyzer_dump_int(res); //expected-warning{{2147483647 S32b}}
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

void test_bool_assign(void)
{
    int res;

    // Reproduce issue from GH#111147. __builtin_*_overflow functions
    // should return _Bool, but not int.
    _Bool ret = __builtin_mul_overflow(10, 20, &res); // no crash
}

void test_unsigned_int128_negative_literal(void)
{
    unsigned __int128 a = 42;

    // This should not crash the static analyzer.
    // Reproduces issue from GitHub #150206 where __builtin_mul_overflow
    // with unsigned __int128 and negative literal caused a crash in
    // SimpleSValBuilder::MakeSymIntVal.
    __builtin_mul_overflow(a, -16, &a); // no crash

    // Test other overflow builtins with the same pattern
    __builtin_add_overflow(a, -16, &a); // no crash
    __builtin_sub_overflow(a, -16, &a); // no crash

    // Test with different negative values
    __builtin_mul_overflow(a, -1, &a);   // no crash
    __builtin_mul_overflow(a, -255, &a); // no crash
}
