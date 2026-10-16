// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,security.ArrayBound,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false -verify %s

// When the checker security.ArrayBound encounters an array subscript operation
// that _may be_ in bounds, it assumes that indexing _is_ in bound. This test
// file validates these assumptions.

void clang_analyzer_value(int);

// Simple case: memory area with a static extent.

extern int FiveInts[5];

void int_plus_one(int len) {
  (void)FiveInts[len + 1]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [-1, 3] }}}
}

void int_neutral(int len) {
  (void)FiveInts[len]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 4] }}}
}

void int_minus_one(int len) {
  (void)FiveInts[len - 1]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [1, 5] }}}
}

void unsigned_plus_one(unsigned len) {
  (void)FiveInts[len + 1]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 3] }}}
}

void unsigned_neutral(unsigned len) {
  (void)FiveInts[len]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 4] }}}
}

void unsigned_minus_one(unsigned len) {
  (void)FiveInts[len - 1]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [1, 5] }}}
}

void ll_plus_one(long long len) {
  (void)FiveInts[len + 1]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [-1, 3] }}}
}

void ll_neutral(long long len) {
  (void)FiveInts[len]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 4] }}}
}

void ll_minus_one(long long len) {
  (void)FiveInts[len - 1]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [1, 5] }}}
}

void ull_plus_one(unsigned long long len) {
  (void)FiveInts[len + 1]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 3] }}}
}

void ull_neutral(unsigned long long len) {
  (void)FiveInts[len]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 4] }}}
}

void ull_minus_one(unsigned long long len) {
  (void)FiveInts[len - 1]; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [1, 5] }}}
}

// Also try the same with a dynamically allocated memory block, because in the
// past there were issues with the type/signedness of dynamic extent symbols.

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

void dyn_int_plus_one(int len) {
  char *p = malloc(5);
  p[len + 1] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [-1, 3] }}}
  free(p);
}

void dyn_int_neutral(int len) {
  char *p = malloc(5);
  p[len] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 4] }}}
  free(p);
}

void dyn_int_minus_one(int len) {
  char *p = malloc(5);
  p[len - 1] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [1, 5] }}}
  free(p);
}

void dyn_unsigned_plus_one(unsigned len) {
  char *p = malloc(5);
  p[len + 1] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 3] }}}
  free(p);
}

void dyn_unsigned_neutral(unsigned len) {
  char *p = malloc(5);
  p[len] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 4] }}}
  free(p);
}

void dyn_unsigned_minus_one(unsigned len) {
  char *p = malloc(5);
  p[len - 1] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [1, 5] }}}
  free(p);
}

void dyn_ll_plus_one(long long len) {
  char *p = malloc(5);
  p[len + 1] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [-1, 3] }}}
  free(p);
}

void dyn_ll_neutral(long long len) {
  char *p = malloc(5);
  p[len] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 4] }}}
  free(p);
}

void dyn_ll_minus_one(long long len) {
  char *p = malloc(5);
  p[len - 1] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [1, 5] }}}
  free(p);
}

void dyn_ull_plus_one(unsigned long long len) {
  char *p = malloc(5);
  p[len + 1] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 3] }}}
  free(p);
}

void dyn_ull_neutral(unsigned long long len) {
  char *p = malloc(5);
  p[len] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [0, 4] }}}
  free(p);
}

void dyn_ull_minus_one(unsigned long long len) {
  char *p = malloc(5);
  p[len - 1] = 1; // no-warning
  clang_analyzer_value(len); // expected-warning {{{ [1, 5] }}}
  free(p);
}
