// RUN: clang-tidy %s -checks=-*,modernize-use-nullptr -- -std=c17 | count 0

// Note: this test expects no diagnostics, but FileCheck cannot handle that,
// hence the use of | count 0.

#define NULL 0
void f(void) {
  char *str = NULL; // ok
  (void)str;
}
