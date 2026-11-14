// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config graph-trim-interval=10 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config graph-trim-interval=15 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config graph-trim-interval=19 -verify %s

// Condition tracking must not use a node reclaimed before the report is built.

void use(int *);

static unsigned elems(unsigned k) {
  switch (k) {
  case 0: return 2;
  case 1: return 4;
  default: return 16;
  }
}

static void reorder(int **in, unsigned n, unsigned s) {
  int *t[8];
  for (unsigned i = 0; i < (n / 16) * s; i += 2)
    t[i / 2] = in[i];
  for (unsigned i = 0; i < s; i++)
    use(t[2 * i]); // expected-warning{{1st function call argument is an uninitialized value}}
}

void top(int **in, unsigned k, unsigned s) {
  reorder(in, elems(k) * 16, s);
}
