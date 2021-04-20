// Test that unused globals are included in the root set.
// RUN: %clangxx_lsan -O2 %s -DTEST_LIB -c -o %t.o
// RUN: %clangxx_lsan -O2 %s %t.o -o %t
// RUN: LSAN_BASE="use_stacks=0:use_registers=0"
// RUN: %env_lsan_opts=$LSAN_BASE:"use_globals=0" not %run %t 2>&1 | FileCheck %s --check-prefixes=LEAK
// RUN: %env_lsan_opts=$LSAN_BASE:"use_globals=1" %run %t 2>&1 | FileCheck %s --implicit-check-not=leak
// RUN: %env_lsan_opts="" %run %t 2>&1 | FileCheck %s --implicit-check-not=leak

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TEST_LIB

void set(char *a) {
  strcpy(a, "hello");
}

#else

static void *g;

void set(char *a);
void foo(void *a) {
  // Store from a different function to suppress global localization.
  g = a;
}

int main() {
  char a[10];
  set(a);
  char *b = strdup(a);
  printf("%p %s\n", b, b);
  g = b;
}

#endif

// LEAK: LeakSanitizer: detected memory leaks
