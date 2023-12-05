// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -Wformat -Wformat-signedness %s

int printf(const char *restrict format, ...);
int scanf(const char * restrict, ...);

void test_printf_bool(_Bool x)
{
    printf("%d", x); // no-warning
    printf("%u", x); // expected-warning{{format specifies type 'unsigned int' but the argument has type '_Bool'}}
    printf("%x", x); // expected-warning{{format specifies type 'unsigned int' but the argument has type '_Bool'}}
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
