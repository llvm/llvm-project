// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// expected-no-diagnostics

void halt() __attribute__((__noreturn__));
void assert(int b) {
  if (!b)
    halt();
}

void decode(unsigned width) {
  assert(width > 0);

  int base;
  bool inited = false;

  int i = 0;

  if (i % width == 0) {
    base = 512;
    inited = true;
  }

  base += 1; // no-warning

  if (base >> 10)
    assert(false);
}
