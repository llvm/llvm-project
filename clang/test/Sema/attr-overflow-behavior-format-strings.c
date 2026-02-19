// RUN: %clang_cc1 -fsyntax-only -verify -Wformat -fexperimental-overflow-behavior-types -isystem %S/Inputs %s

// Test format string checking with overflow behavior types
// This ensures that OverflowBehaviorTypes work seamlessly with printf/scanf
// without spurious format warnings.

int printf(const char *restrict, ...);
int sprintf(char *restrict, const char *restrict, ...);
int snprintf(char *restrict, __SIZE_TYPE__, const char *restrict, ...);

int scanf(const char *restrict, ...);
int sscanf(const char *restrict, const char *restrict, ...);

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __trap __attribute__((overflow_behavior(trap)))

typedef int __ob_wrap wrap_int;
typedef int __ob_trap no_trap_int;
typedef unsigned int __ob_wrap wrap_uint;
typedef unsigned int __ob_trap no_trap_uint;
typedef short __ob_wrap wrap_short;
typedef long __ob_trap no_trap_long;

void test_printf_compatibility() {
  wrap_int wi = 42;
  no_trap_int ni = 42;
  wrap_uint wu = 42U;
  no_trap_uint nu = 42U;
  wrap_short ws = 42;
  no_trap_long nl = 42L;

  // These should all work without warnings - OBTs should be treated as their underlying types
  printf("%d", wi);
  printf("%d", ni);
  printf("%u", wu);
  printf("%u", nu);
  printf("%hd", ws);
  printf("%ld", nl);
  printf("%x", wu);
  printf("%o", nu);

  printf("%s", wi);     // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  printf("%f", wi);     // expected-warning{{format specifies type 'double' but the argument has type 'int'}}
  printf("%ld", wi);    // expected-warning{{format specifies type 'long' but the argument has type 'int'}}
}

void test_scanf_compatibility() {
  wrap_int wi;
  no_trap_int ni;
  wrap_uint wu;
  no_trap_uint nu;
  wrap_short ws;
  no_trap_long nl;

  // These should all work without warnings - pointers to OBTs should be treated as pointers to underlying types
  scanf("%d", &wi);
  scanf("%d", &ni);
  scanf("%u", &wu);
  scanf("%u", &nu);
  scanf("%hd", &ws);
  scanf("%ld", &nl);
  scanf("%x", &wu);
  scanf("%o", &nu);

  scanf("%s", &wi);     // expected-warning{{format specifies type 'char *' but the argument has type 'wrap_int *'}}
  scanf("%f", &wi);     // expected-warning{{format specifies type 'float *' but the argument has type 'wrap_int *'}}
  scanf("%ld", &wi);    // expected-warning{{format specifies type 'long *' but the argument has type 'wrap_int *'}}
}

void test_mixed_formats() {
  wrap_int wi = 42;
  int regular_int = 42;

  printf("%d + %d = %d", wi, regular_int, wi + regular_int);
  scanf("%d %d", &wi, &regular_int);
}

typedef unsigned char __ob_wrap wrap_byte;
typedef long long __ob_trap safe_longlong;

void test_typedef_formats() {
  wrap_byte wb = 255;
  safe_longlong sll = 123456789LL;

  printf("%hhu", wb);    // OK: wrap_byte -> unsigned char for %hhu
  printf("%lld", sll);   // OK: safe_longlong -> long long for %lld

  scanf("%hhu", &wb);    // OK: &wb treated as unsigned char* for %hhu
  scanf("%lld", &sll);   // OK: &sll treated as long long* for %lld

  printf("%d", wb);      // OK: wb implicitly cast to int
  printf("%d", sll);     // expected-warning{{format specifies type 'int' but the argument has type 'long long'}}
}
