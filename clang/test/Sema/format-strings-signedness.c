// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -std=c11 -fsyntax-only -verify -Wformat -Wformat-signedness %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -std=c11 -fsyntax-only -verify -Wformat -Wformat-signedness %s

// Verify that -Wformat-signedness alone (without -Wformat) trigger the
// warnings. Note in gcc this will not trigger the signedness warnings as
// -Wformat is default off in gcc.
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -std=c11 -fsyntax-only -verify -Wformat-signedness %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -std=c11 -fsyntax-only -verify -Wformat-signedness %s

// Verify that -Wformat-signedness warnings are not reported with only -Wformat
// (gcc compat).
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -std=c11 -fsyntax-only -Wformat -verify=okay %s

// Verify that -Wformat-signedness with -Wno-format are not reported (gcc compat).
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -std=c11 -fsyntax-only -Wformat-signedness -Wno-format -verify=okay %s
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -std=c11 -fsyntax-only -Wno-format -Wformat-signedness -verify=okay %s
// okay-no-diagnostics

int printf(const char *restrict format, ...);
int scanf(const char * restrict, ...);

void test_printf_bool(_Bool x)
{
    printf("%d", x); // no-warning
    printf("%u", x); // no-warning
    printf("%x", x); // no-warning
}

void test_printf_char(char x)
{
    printf("%c", x); // no-warning
}

void test_printf_unsigned_char(unsigned char x)
{
    printf("%c", x); // no-warning
}

void test_printf_int(int x)
{
    printf("%d", x); // no-warning
    printf("%u", x); // expected-warning{{format specifies type 'unsigned int' but the argument has type 'int'}}
    printf("%x", x); // expected-warning{{format specifies type 'unsigned int' but the argument has type 'int'}}
}

void test_printf_unsigned(unsigned x)
{
    printf("%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'unsigned int'}}
    printf("%u", x); // no-warning
    printf("%x", x); // no-warning
}

void test_printf_long(long x)
{
    printf("%ld", x); // no-warning
    printf("%lu", x); // expected-warning{{format specifies type 'unsigned long' but the argument has type 'long'}}
    printf("%lx", x); // expected-warning{{format specifies type 'unsigned long' but the argument has type 'long'}}
}

void test_printf_unsigned_long(unsigned long x)
{
    printf("%ld", x); // expected-warning{{format specifies type 'long' but the argument has type 'unsigned long'}}
    printf("%lu", x); // no-warning
    printf("%lx", x); // no-warning
}

void test_printf_long_long(long long x)
{
    printf("%lld", x); // no-warning
    printf("%llu", x); // expected-warning{{format specifies type 'unsigned long long' but the argument has type 'long long'}}
    printf("%llx", x); // expected-warning{{format specifies type 'unsigned long long' but the argument has type 'long long'}}
}

void test_printf_unsigned_long_long(unsigned long long x)
{
    printf("%lld", x); // expected-warning{{format specifies type 'long long' but the argument has type 'unsigned long long'}}
    printf("%llu", x); // no-warning
    printf("%llx", x); // no-warning
}

enum enum_int {
    minus_1 = -1
};

void test_printf_enum_int(enum enum_int x)
{
    printf("%d", x); // no-warning
    printf("%u", x); // expected-warning{{format specifies type 'unsigned int' but the argument has underlying type 'int'}}
    printf("%x", x); // expected-warning{{format specifies type 'unsigned int' but the argument has underlying type 'int'}}
}

#ifndef _WIN32 // Disabled due to enums have different underlying type on _WIN32
enum enum_unsigned {
    zero = 0
};

void test_printf_enum_unsigned(enum enum_unsigned x)
{
    printf("%d", x); // expected-warning{{format specifies type 'int' but the argument has underlying type 'unsigned int'}}
    printf("%u", x); // no-warning
    printf("%x", x); // no-warning
}

enum enum_long {
    minus_one = -1,
    int_val = __INT_MAX__, // INT_MAX
    unsigned_val = (unsigned)(-__INT_MAX__ -1) // (unsigned)INT_MIN
};

void test_printf_enum_long(enum enum_long x)
{
    printf("%ld", x); // no-warning
    printf("%lu", x); // expected-warning{{format specifies type 'unsigned long' but the argument has underlying type 'long'}}
    printf("%lx", x); // expected-warning{{format specifies type 'unsigned long' but the argument has underlying type 'long'}}
}

enum enum_unsigned_long {
    uint_max_plus = (unsigned long)(__INT_MAX__ *2U +1U)+1, // (unsigned long)UINT_MAX+1
};

void test_printf_enum_unsigned_long(enum enum_unsigned_long x)
{
    printf("%ld", x); // expected-warning{{format specifies type 'long' but the argument has underlying type 'unsigned long'}}
    printf("%lu", x); // no-warning
    printf("%lx", x); // no-warning
}
#endif

void test_scanf_char(char *y) {
  scanf("%c", y); // no-warning
}

void test_scanf_unsigned_char(unsigned char *y) {
  scanf("%c", y); // no-warning
}

void test_scanf_int(int *x) {
  scanf("%d", x); // no-warning
  scanf("%u", x); // expected-warning{{format specifies type 'unsigned int *' but the argument has type 'int *'}}
  scanf("%x", x); // expected-warning{{format specifies type 'unsigned int *' but the argument has type 'int *'}}
}

void test_scanf_unsigned(unsigned *x) {
  scanf("%d", x); // expected-warning{{format specifies type 'int *' but the argument has type 'unsigned int *'}}
  scanf("%u", x); // no-warning
  scanf("%x", x); // no-warning
}

void test_scanf_long(long *x) {
  scanf("%ld", x); // no-warning
  scanf("%lu", x); // expected-warning{{format specifies type 'unsigned long *' but the argument has type 'long *'}}
  scanf("%lx", x); // expected-warning{{format specifies type 'unsigned long *' but the argument has type 'long *'}}
}

void test_scanf_unsigned_long(unsigned long *x) {
  scanf("%ld", x); // expected-warning{{format specifies type 'long *' but the argument has type 'unsigned long *'}}
  scanf("%lu", x); // no-warning
  scanf("%lx", x); // no-warning
}

void test_scanf_longlong(long long *x) {
  scanf("%lld", x); // no-warning
  scanf("%llu", x); // expected-warning{{format specifies type 'unsigned long long *' but the argument has type 'long long *'}}
  scanf("%llx", x); // expected-warning{{format specifies type 'unsigned long long *' but the argument has type 'long long *'}}
}

void test_scanf_unsigned_longlong(unsigned long long *x) {
  scanf("%lld", x); // expected-warning{{format specifies type 'long long *' but the argument has type 'unsigned long long *'}}
  scanf("%llu", x); // no-warning
  scanf("%llx", x); // no-warning
}

void test_scanf_enum_int(enum enum_int *x) {
  scanf("%d", x); // no-warning
  scanf("%u", x); // expected-warning{{format specifies type 'unsigned int *' but the argument has type 'enum enum_int *'}}
  scanf("%x", x); // expected-warning{{format specifies type 'unsigned int *' but the argument has type 'enum enum_int *'}}
}

#ifndef _WIN32 // Disabled due to enums have different underlying type on _WIN32
void test_scanf_enum_unsigned(enum enum_unsigned *x) {
  scanf("%d", x); // expected-warning{{format specifies type 'int *' but the argument has type 'enum enum_unsigned *'}}
  scanf("%u", x); // no-warning
  scanf("%x", x); // no-warning
}

void test_scanf_enum_long(enum enum_long *x) {
  scanf("%ld", x); // no-warning
  scanf("%lu", x); // expected-warning{{format specifies type 'unsigned long *' but the argument has type 'enum enum_long *'}}
  scanf("%lx", x); // expected-warning{{format specifies type 'unsigned long *' but the argument has type 'enum enum_long *'}}
}

void test_scanf_enum_unsigned_long(enum enum_unsigned_long *x) {
  scanf("%ld", x); // expected-warning{{format specifies type 'long *' but the argument has type 'enum enum_unsigned_long *'}}
  scanf("%lu", x); // no-warning
  scanf("%lx", x); // no-warning
}
#endif

// Verify that we get no warnings from <inttypes.h>

typedef short int int16_t;
typedef unsigned short int uint16_t;

void test_printf_priX16(int16_t x) {
  printf("PRId16: %" "d" /*PRId16*/ "\n", x); // no-warning
  printf("PRIi16: %" "i" /*PRIi16*/ "\n", x); // no-warning
}

void test_printf_unsigned_priX16(uint16_t x) {
  printf("PRIo16: %" "o" /*PRIo16*/ "\n", x); // no-warning
  printf("PRIu16: %" "u" /*PRIu16*/ "\n", x); // no-warning
  printf("PRIx16: %" "x" /*PRIx16*/ "\n", x); // no-warning
  printf("PRIX16: %" "X" /*PRIX16*/ "\n", x); // no-warning
}

// Verify that we can suppress a -Wformat-signedness warning by ignoring
// -Wformat (gcc compat).
void test_suppress(int x)
{
#pragma GCC diagnostic ignored "-Wformat"
    printf("%u", x);
}
