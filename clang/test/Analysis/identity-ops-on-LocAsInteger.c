// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core,unix.Malloc,debug.ExprInspection

// Identity operations on a LocAsInteger must preserve the original location.
// GH#220972.

typedef unsigned __INTPTR_TYPE__ uintptr_t;
typedef __SIZE_TYPE__ size_t;

void *malloc(size_t);
void free(void *);
void clang_analyzer_eval(int);

void test_add_zero(void)
{
  void *ptr = malloc(16);
  if (ptr == 0)
    return;

  uintptr_t value = (uintptr_t)ptr;
  value += 0;
  clang_analyzer_eval(value == (uintptr_t)ptr); // expected-warning{{TRUE}}

  free((void *)value); // no-warning
}

void test_other_identity_ops(void)
{
  void *ptr = malloc(16);
  if (ptr == 0)
    return;

  uintptr_t value = (uintptr_t)ptr;
  value -= 0;
  value |= 0;
  value ^= 0;
  value <<= 0;
  value >>= 0;
  value *= 1;
  value /= 1;
  clang_analyzer_eval(value == (uintptr_t)ptr); // expected-warning{{TRUE}}

  free((void *)value); // no-warning
}

void test_spelled_out_and_swapped(void)
{
  void *ptr = malloc(16);
  if (ptr == 0)
    return;

  uintptr_t value = (uintptr_t)ptr;
  value = value + 0;
  value = 0 + value;
  value = 0 | value;
  clang_analyzer_eval(value == (uintptr_t)ptr); // expected-warning{{TRUE}}

  free((void *)value); // no-warning
}

void test_narrower_result(void)
{
  void *ptr = malloc(16);
  if (ptr == 0)
    return;

  unsigned value = (unsigned)(uintptr_t)ptr;
  value += 0;
  clang_analyzer_eval(value == (unsigned)(uintptr_t)ptr); // expected-warning{{TRUE}}

  free(ptr); // no-warning
}
