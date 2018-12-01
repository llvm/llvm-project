// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,debug.ExprInspection %s -std=c++11 -verify

void clang_analyzer_eval(bool);
void clang_analyzer_warnIfReached();

void testAddressof(int x) {
  clang_analyzer_eval(&x == __builtin_addressof(x)); // expected-warning{{TRUE}}
}

void testSize() {
  struct {
    int x;
    int y;
    char z;
  } object;
  clang_analyzer_eval(__builtin_object_size(&object.y, 0) == sizeof(object) - sizeof(int)); // expected-warning{{TRUE}}

  // Clang can't actually evaluate these builtin "calls", but importantly they don't actually evaluate the argument expression either.
  int i = 0;
  char buf[10];
  clang_analyzer_eval(__builtin_object_size(&buf[i++], 0) == sizeof(buf)); // expected-warning{{FALSE}}
  clang_analyzer_eval(__builtin_object_size(&buf[++i], 0) == sizeof(buf) - 1); // expected-warning{{FALSE}}

  clang_analyzer_eval(i == 0); // expected-warning{{TRUE}}
}

void test_assume_aligned_1(char *p) {
  char *q;

  q = (char*) __builtin_assume_aligned(p, 16);
  clang_analyzer_eval(p == q); // expected-warning{{TRUE}}
}

void test_assume_aligned_2(char *p) {
  char *q;

  q = (char*) __builtin_assume_aligned(p, 16, 8);
  clang_analyzer_eval(p == q); // expected-warning{{TRUE}}
}

void test_assume_aligned_3(char *p) {
  void *q;

  q = __builtin_assume_aligned(p, 16, 8);
  clang_analyzer_eval(p == q); // expected-warning{{TRUE}}
}

void test_assume_aligned_4(char *p) {
  char *q;

  q = (char*) __builtin_assume_aligned(p + 1, 16);
  clang_analyzer_eval(p == q); // expected-warning{{FALSE}}
}

void f(int i) {
  __builtin_assume(i < 10);
  clang_analyzer_eval(i < 15); // expected-warning {{TRUE}}
}

void g(int i) {
  if (i > 5) {
    __builtin_assume(i < 5);
    clang_analyzer_warnIfReached(); // Assumtion contradicts constraints.
                                    // We give up the analysis on this path.
  }
}

void test_constant_p() {
  int i = 1;
  const int j = 2;
  constexpr int k = 3;
  clang_analyzer_eval(__builtin_constant_p(42) == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(__builtin_constant_p(i) == 0); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(__builtin_constant_p(j) == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(__builtin_constant_p(k) == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(__builtin_constant_p(i + 42) == 0); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(__builtin_constant_p(j + 42) == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(__builtin_constant_p(k + 42) == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(__builtin_constant_p(" ") == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(__builtin_constant_p(test_constant_p) == 0); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(__builtin_constant_p(k - 3) == 0); // expected-warning {{FALSE}}
  clang_analyzer_eval(__builtin_constant_p(k - 3) == 1); // expected-warning {{TRUE}}
}
