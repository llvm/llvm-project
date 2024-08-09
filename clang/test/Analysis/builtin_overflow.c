// RUN: %clang_analyze_cc1 -triple x86_64-unknown-unknown -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection

#define NULL ((void *)0)
#define INT_MAX __INT_MAX__

void clang_analyzer_dump_int(int);

void test1(void)
{
   int res;

   if (__builtin_add_overflow(10, 20, &res)) {
     clang_analyzer_dump_int(res); //expected-warning{{1st function call argument is an uninitialized value}}
     return;
   }

   clang_analyzer_dump_int(res); //expected-warning{{30}}
}

void test2(void)
{
   int res;

   __builtin_add_overflow(10, 20, &res);
   clang_analyzer_dump_int(res); //expected-warning{{1st function call argument is an uninitialized value}} expected-warning{{S32b}}
}

void test3(void)
{
   int res;

   if (__builtin_sub_overflow(10, 20, &res)) {
     clang_analyzer_dump_int(res); //expected-warning{{1st function call argument is an uninitialized value}}
     return;
   }

   clang_analyzer_dump_int(res); //expected-warning{{-10}}
}

void test4(void)
{
   int res;

   if (__builtin_sub_overflow(10, 20, &res)) {
     clang_analyzer_dump_int(res); //expected-warning{{1st function call argument is an uninitialized value}}
     return;
   }

   if (res != -10) {
     *(volatile char *)NULL; //no warning
   }
}

void test5(void)
{
   int res;

   if (__builtin_mul_overflow(10, 20, &res)) {
     clang_analyzer_dump_int(res); //expected-warning{{1st function call argument is an uninitialized value}}
     return;
   }

   clang_analyzer_dump_int(res); //expected-warning{{200}}
}
