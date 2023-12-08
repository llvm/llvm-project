// RUN: cp %s %t
// RUN: %clang_cc1 -Wformat -Wformat-signedness -fixit %t
// RUN: %clang_cc1 -fsyntax-only -Wformat -Wformat-signedness -Werror %t
// RUN: %clang_cc1 -E -o - %t | FileCheck %s

#include <limits.h>

int printf(const char *restrict format, ...);

enum foo {
    minus_one = -1,
    int_val = INT_MAX,
    unsigned_val = (unsigned)INT_MIN
};

void test_printf_int(int x)
{
    printf("%u", x);
}

void test_printf_unsigned(unsigned x)
{
    printf("%d", x);
}

void test_printf_long(long x)
{
    printf("%lu", x);
}

void test_printf_unsigned_long(unsigned long x)
{
    printf("%ld", x);
}

void test_printf_long_long(long long x)
{
    printf("%llu", x);
}

void test_printf_unsigned_long_long(unsigned long long x)
{
    printf("%lld", x);
}

void test_printf_enum(enum foo x)
{
    printf("%lu", x);
}

// Validate the fixes.
// CHECK: void test_printf_int(int x)
// CHECK: printf("%d", x);
// CHECK: void test_printf_unsigned(unsigned x)
// CHECK: printf("%u", x);
// CHECK: void test_printf_long(long x)
// CHECK: printf("%ld", x);
// CHECK: void test_printf_unsigned_long(unsigned long x)
// CHECK: printf("%lu", x);
// CHECK: void test_printf_long_long(long long x)
// CHECK: printf("%lld", x);
// CHECK: void test_printf_unsigned_long_long(unsigned long long x)
// CHECK: printf("%llu", x);
// CHECK: void test_printf_enum(enum foo x)
// CHECK: printf("%ld", x);
