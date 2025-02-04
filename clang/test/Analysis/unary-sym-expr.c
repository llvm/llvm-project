// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify

void clang_analyzer_eval(int);
void clang_analyzer_dump(int);

int test(int x, int y) {

  clang_analyzer_dump(-x);       // expected-warning{{-reg_$0<int x>}}
  clang_analyzer_dump(~x);       // expected-warning{{~reg_$0<int x>}}
  int z = x + y;
  clang_analyzer_dump(-z);       // expected-warning{{-((reg_$0<int x>) + (reg_$3<int y>))}}
  clang_analyzer_dump(-(x + y)); // expected-warning{{-((reg_$0<int x>) + (reg_$3<int y>))}}
  clang_analyzer_dump(-x + y);   // expected-warning{{(-reg_$0<int x>) + (reg_$3<int y>)}}

  if (-x == 0) {
    clang_analyzer_eval(-x == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(-x > 0);  // expected-warning{{FALSE}}
    clang_analyzer_eval(-x < 0);  // expected-warning{{FALSE}}
  }
  if (~y == 0) {
    clang_analyzer_eval(~y == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(~y > 0);  // expected-warning{{FALSE}}
    clang_analyzer_eval(~y < 0);  // expected-warning{{FALSE}}
  }
  (void)(x);
  return 42;
}

void test_svalbuilder_simplification_add(int x, int y) {
  if (x + y != 3)
    return;
  clang_analyzer_eval(-(x + y) == -3); // expected-warning{{TRUE}}
  clang_analyzer_eval(-(y + x) == -3); // expected-warning{{TRUE}}
}

void test_svalbuilder_simplification_mul(int x, int y) {
  if (x * y != 3)
    return;
  clang_analyzer_eval(-(x * y) == -3); // expected-warning{{TRUE}}
  clang_analyzer_eval(-(y * x) == -3); // expected-warning{{TRUE}}
}

void test_svalbuilder_simplification_and(int x, int y) {
  if ((x & y) != 3)
    return;
  clang_analyzer_eval(-(x & y) == -3); // expected-warning{{TRUE}}
  clang_analyzer_eval(-(y & x) == -3); // expected-warning{{TRUE}}
}

void test_svalbuilder_simplification_or(int x, int y) {
  if ((x | y) != 3)
    return;
  clang_analyzer_eval(-(x | y) == -3); // expected-warning{{TRUE}}
  clang_analyzer_eval(-(y | x) == -3); // expected-warning{{TRUE}}
}

void test_svalbuilder_simplification_xor(int x, int y) {
  if ((x ^ y) != 3)
    return;
  clang_analyzer_eval(-(x ^ y) == -3); // expected-warning{{TRUE}}
  clang_analyzer_eval(-(y ^ x) == -3); // expected-warning{{TRUE}}
}

int test_fp(int flag) {
  int value;
  if (flag == 0)
    value = 1;
  if (-flag == 0)
    return value; // no-warning
  return 42;
}
