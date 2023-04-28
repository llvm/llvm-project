// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.security.ArrayBoundV2,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false -verify %s

void clang_analyzer_eval(int);
void clang_analyzer_printState(void);

typedef unsigned long long size_t;
const char a[] = "abcd"; // extent: 5 bytes

void symbolic_size_t_and_int0(size_t len) {
  (void)a[len + 1]; // no-warning
  // We infered that the 'len' must be in a specific range to make the previous indexing valid.
  // len: [0,3]
  clang_analyzer_eval(len <= 3); // expected-warning {{TRUE}}
  clang_analyzer_eval(len <= 2); // expected-warning {{UNKNOWN}}
}

void symbolic_size_t_and_int1(size_t len) {
  (void)a[len]; // no-warning
  // len: [0,4]
  clang_analyzer_eval(len <= 4); // expected-warning {{TRUE}}
  clang_analyzer_eval(len <= 3); // expected-warning {{UNKNOWN}}
}

void symbolic_size_t_and_int2(size_t len) {
  (void)a[len - 1]; // no-warning
  // len: [1,5]
  clang_analyzer_eval(1 <= len && len <= 5); // expected-warning {{TRUE}}
  clang_analyzer_eval(2 <= len);             // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(len <= 4);             // expected-warning {{UNKNOWN}}
}

void symbolic_uint_and_int0(unsigned len) {
  (void)a[len + 1]; // no-warning
  // len: [0,3]
  clang_analyzer_eval(0 <= len && len <= 3); // expected-warning {{TRUE}}
  clang_analyzer_eval(1 <= len);             // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(len <= 2);             // expected-warning {{UNKNOWN}}
}

void symbolic_uint_and_int1(unsigned len) {
  (void)a[len]; // no-warning
  // len: [0,4]
  clang_analyzer_eval(0 <= len && len <= 4); // expected-warning {{TRUE}}
  clang_analyzer_eval(1 <= len);             // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(len <= 3);             // expected-warning {{UNKNOWN}}
}
void symbolic_uint_and_int2(unsigned len) {
  (void)a[len - 1]; // no-warning
  // len: [1,5]
  clang_analyzer_eval(1 <= len && len <= 5); // expected-warning {{TRUE}}
  clang_analyzer_eval(2 <= len);             // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(len <= 4);             // expected-warning {{UNKNOWN}}
}

void symbolic_int_and_int0(int len) {
  (void)a[len + 1]; // no-warning
  // len: [-1,3]
  clang_analyzer_eval(-1 <= len && len <= 3); // expected-warning {{TRUE}}
  clang_analyzer_eval(0 <= len);              // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(len <= 2);              // expected-warning {{UNKNOWN}}
}
void symbolic_int_and_int1(int len) {
  (void)a[len]; // no-warning
  // len: [0,4]
  clang_analyzer_eval(0 <= len && len <= 4); // expected-warning {{TRUE}}
  clang_analyzer_eval(1 <= len);             // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(len <= 3);             // expected-warning {{UNKNOWN}}
}
void symbolic_int_and_int2(int len) {
  (void)a[len - 1]; // no-warning
  // len: [1,5]
  clang_analyzer_eval(1 <= len && len <= 5); // expected-warning {{TRUE}}
  clang_analyzer_eval(2 <= len);             // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(len <= 4);             // expected-warning {{UNKNOWN}}
}

void symbolic_longlong_and_int0(long long len) {
  (void)a[len + 1]; // no-warning
  // len: [-1,3]
  clang_analyzer_eval(-1 <= len && len <= 3); // expected-warning {{TRUE}}
  clang_analyzer_eval(0 <= len);              // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(len <= 2);              // expected-warning {{UNKNOWN}}
}

void symbolic_longlong_and_int1(long long len) {
  (void)a[len]; // no-warning
  // len: [0,4]
  clang_analyzer_eval(0 <= len && len <= 4); // expected-warning {{TRUE}}
  clang_analyzer_eval(1 <= len);             // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(len <= 3);             // expected-warning {{UNKNOWN}}
}

void symbolic_longlong_and_int2(long long len) {
  (void)a[len - 1]; // no-warning
  // len: [1,5]
  clang_analyzer_eval(1 <= len && len <= 5); // expected-warning {{TRUE}}
  clang_analyzer_eval(2 <= len);             // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(len <= 4);             // expected-warning {{UNKNOWN}}
}
