// Test __sanitizer_reset_coverage().

// RUN: %clangxx_asan -fsanitize-coverage=func %s -o %t
// RUN: %env_asan_opts=coverage=1 %run %t

// https://github.com/google/sanitizers/issues/618
// UNSUPPORTED: android

#include <sanitizer/coverage_interface.h>
#include <stdio.h>
#include <assert.h>
static volatile int sink;
__attribute__((noinline)) void bar() { sink = 2; }
__attribute__((noinline)) void foo() { sink = 1; }

// In MSVC 2015, printf is an inline function, which causes this test to fail as
// it introduces an extra coverage point. Define away printf on that platform to
// avoid the issue.
#if _MSC_VER >= 1900
# define printf(arg, ...)
#endif

#define GET_AND_PRINT_COVERAGE()                                       \
  bitset = 0;                                                  \
  for (size_t i = 0; i < n_guards; i++)                        \
    if (guards[i]) bitset |= 1U << i;                          \
  printf("line %d: bitset %zd total: %zd\n", __LINE__, bitset, \
         __sanitizer_get_total_unique_coverage());

#define IS_POWER_OF_TWO(a) ((a & ((a) - 1)) == 0)

int main() {
  size_t *guards = 0;
  size_t bitset;
  size_t n_guards = __sanitizer_get_coverage_guards(&guards);

  GET_AND_PRINT_COVERAGE();
  size_t main_bit = bitset;
  assert(IS_POWER_OF_TWO(main_bit));

  foo();
  GET_AND_PRINT_COVERAGE();
  size_t foo_bit = bitset & ~main_bit;
  assert(IS_POWER_OF_TWO(foo_bit));

  bar();
  GET_AND_PRINT_COVERAGE();
  size_t bar_bit = bitset & ~(main_bit | foo_bit);
  assert(IS_POWER_OF_TWO(bar_bit));

  __sanitizer_reset_coverage();
  assert(__sanitizer_get_total_unique_coverage() == 0);
  GET_AND_PRINT_COVERAGE();
  assert(bitset == 0);

  foo();
  GET_AND_PRINT_COVERAGE();
  assert(bitset == foo_bit);

  bar();
  GET_AND_PRINT_COVERAGE();
  assert(bitset == (foo_bit | bar_bit));
}
