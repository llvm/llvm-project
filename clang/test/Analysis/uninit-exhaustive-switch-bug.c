// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// expected-no-diagnostics

int rand(void);

void test(void) {
  int offset = 0;
  int value;
  int test = rand();
  switch (test & 0x1) {
  case 0:
  case 1:
    value = 0;
    break;
  }

  offset += value; // no-warning
}
