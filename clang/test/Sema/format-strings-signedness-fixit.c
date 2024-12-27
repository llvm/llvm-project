// RUN: cp %s %t
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -Wformat -Wformat-signedness -fixit %t
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -Wformat -Wformat-signedness -Werror %t
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -E -o - %t | FileCheck %s

#include <limits.h>

int printf(const char *restrict format, ...);

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

enum enum_int {
    minus_1 = -1
};

void test_printf_enum_int(enum enum_int x)
{
    printf("%u", x);
}

enum enum_unsigned {
    zero = 0
};

void test_printf_enum_unsigned(enum enum_unsigned x)
{
    printf("%d", x);
}

enum enum_long {
    minus_one = -1,
    int_val = INT_MAX,
    unsigned_val = (unsigned)INT_MIN
};

void test_printf_enum_long(enum enum_long x)
{
    printf("%lu", x);
}

enum enum_unsigned_long {
    uint_max_plus = (unsigned long)UINT_MAX+1,
};

void test_printf_enum_unsigned_long(enum enum_unsigned_long x)
{
    printf("%ld", x);
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
// CHECK: void test_printf_enum_int(enum enum_int x)
// CHECK: printf("%d", x);
// CHECK: void test_printf_enum_unsigned(enum enum_unsigned x)
// CHECK: printf("%u", x);
// CHECK: void test_printf_enum_long(enum enum_long x)
// CHECK: printf("%ld", x);
// CHECK: void test_printf_enum_unsigned_long(enum enum_unsigned_long x)
// CHECK: printf("%lu", x);
