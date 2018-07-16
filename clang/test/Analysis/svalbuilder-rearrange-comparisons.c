// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection,core.builtin -analyzer-config aggressive-relational-comparison-simplification=true -verify %s

void clang_analyzer_dump(int x);
void clang_analyzer_eval(int x);

void exit(int);

#define UINT_MAX (~0U)
#define INT_MAX (UINT_MAX & (UINT_MAX >> 1))

extern void __assert_fail (__const char *__assertion, __const char *__file,
    unsigned int __line, __const char *__function)
     __attribute__ ((__noreturn__));
#define assert(expr) \
  ((expr)  ? (void)(0)  : __assert_fail (#expr, __FILE__, __LINE__, __func__))

int g();
int f() {
  int x = g();
  // Assert that no overflows occur in this test file.
  // Assuming that concrete integers are also within that range.
  assert(x <= ((int)INT_MAX / 4));
  assert(x >= -((int)INT_MAX / 4));
  return x;
}

void compare_different_symbol_equal() {
  int x = f(), y = f();
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) == 0}}
}

void compare_different_symbol_plus_left_int_equal() {
  int x = f()+1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) == 1}}
}

void compare_different_symbol_minus_left_int_equal() {
  int x = f()-1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) == 1}}
}

void compare_different_symbol_plus_right_int_equal() {
  int x = f(), y = f()+2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 2}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) == 2}}
}

void compare_different_symbol_minus_right_int_equal() {
  int x = f(), y = f()-2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 2}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) == 2}}
}

void compare_different_symbol_plus_left_plus_right_int_equal() {
  int x = f()+2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) == 1}}
}

void compare_different_symbol_plus_left_minus_right_int_equal() {
  int x = f()+2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) == 3}}
}

void compare_different_symbol_minus_left_plus_right_int_equal() {
  int x = f()-2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) == 3}}
}

void compare_different_symbol_minus_left_minus_right_int_equal() {
  int x = f()-2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) == 1}}
}

void compare_same_symbol_equal() {
  int x = f(), y = x;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_plus_left_int_equal() {
  int x = f(), y = x;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_minus_left_int_equal() {
  int x = f(), y = x;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_plus_right_int_equal() {
  int x = f(), y = x+1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_minus_right_int_equal() {
  int x = f(), y = x-1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_plus_left_plus_right_int_equal() {
  int x = f(), y = x+1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_plus_left_minus_right_int_equal() {
  int x = f(), y = x-1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_minus_left_plus_right_int_equal() {
  int x = f(), y = x+1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_minus_left_minus_right_int_equal() {
  int x = f(), y = x-1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{TRUE}}
}

void compare_different_symbol_less_or_equal() {
  int x = f(), y = f();
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) <= 0}}
}

void compare_different_symbol_plus_left_int_less_or_equal() {
  int x = f()+1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) >= 1}}
}

void compare_different_symbol_minus_left_int_less_or_equal() {
  int x = f()-1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) <= 1}}
}

void compare_different_symbol_plus_right_int_less_or_equal() {
  int x = f(), y = f()+2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 2}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) <= 2}}
}

void compare_different_symbol_minus_right_int_less_or_equal() {
  int x = f(), y = f()-2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 2}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) >= 2}}
}

void compare_different_symbol_plus_left_plus_right_int_less_or_equal() {
  int x = f()+2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) >= 1}}
}

void compare_different_symbol_plus_left_minus_right_int_less_or_equal() {
  int x = f()+2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) >= 3}}
}

void compare_different_symbol_minus_left_plus_right_int_less_or_equal() {
  int x = f()-2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) <= 3}}
}

void compare_different_symbol_minus_left_minus_right_int_less_or_equal() {
  int x = f()-2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) <= 1}}
}

void compare_same_symbol_less_or_equal() {
  int x = f(), y = x;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_plus_left_int_less_or_equal() {
  int x = f(), y = x;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_minus_left_int_less_or_equal() {
  int x = f(), y = x;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_plus_right_int_less_or_equal() {
  int x = f(), y = x+1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_minus_right_int_less_or_equal() {
  int x = f(), y = x-1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_plus_left_plus_right_int_less_or_equal() {
  int x = f(), y = x+1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_plus_left_minus_right_int_less_or_equal() {
  int x = f(), y = x-1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_minus_left_plus_right_int_less_or_equal() {
  int x = f(), y = x+1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_minus_left_minus_right_int_less_or_equal() {
  int x = f(), y = x-1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{TRUE}}
}

void compare_different_symbol_less() {
  int x = f(), y = f();
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) < 0}}
}

void compare_different_symbol_plus_left_int_less() {
  int x = f()+1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) > 1}}
}

void compare_different_symbol_minus_left_int_less() {
  int x = f()-1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) < 1}}
}

void compare_different_symbol_plus_right_int_less() {
  int x = f(), y = f()+2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 2}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) < 2}}
}

void compare_different_symbol_minus_right_int_less() {
  int x = f(), y = f()-2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 2}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) > 2}}
}

void compare_different_symbol_plus_left_plus_right_int_less() {
  int x = f()+2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) > 1}}
}

void compare_different_symbol_plus_left_minus_right_int_less() {
  int x = f()+2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) > 3}}
}

void compare_different_symbol_minus_left_plus_right_int_less() {
  int x = f()-2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) < 3}}
}

void compare_different_symbol_minus_left_minus_right_int_less() {
  int x = f()-2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) < 1}}
}

void compare_same_symbol_less() {
  int x = f(), y = x;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_plus_left_int_less() {
  int x = f(), y = x;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_minus_left_int_less() {
  int x = f(), y = x;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_plus_right_int_less() {
  int x = f(), y = x+1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_minus_right_int_less() {
  int x = f(), y = x-1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_plus_left_plus_right_int_less() {
  int x = f(), y = x+1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_plus_left_minus_right_int_less() {
  int x = f(), y = x-1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_minus_left_plus_right_int_less() {
  int x = f(), y = x+1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_minus_left_minus_right_int_less() {
  int x = f(), y = x-1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{FALSE}}
}

void compare_different_symbol_equal_unsigned() {
  unsigned x = f(), y = f();
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) == 0}}
}

void compare_different_symbol_plus_left_int_equal_unsigned() {
  unsigned x = f()+1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) == 1}}
}

void compare_different_symbol_minus_left_int_equal_unsigned() {
  unsigned x = f()-1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) == 1}}
}

void compare_different_symbol_plus_right_int_equal_unsigned() {
  unsigned x = f(), y = f()+2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 2}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) == 2}}
}

void compare_different_symbol_minus_right_int_equal_unsigned() {
  unsigned x = f(), y = f()-2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 2}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) == 2}}
}

void compare_different_symbol_plus_left_plus_right_int_equal_unsigned() {
  unsigned x = f()+2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) == 1}}
}

void compare_different_symbol_plus_left_minus_right_int_equal_unsigned() {
  unsigned x = f()+2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) == 3}}
}

void compare_different_symbol_minus_left_plus_right_int_equal_unsigned() {
  unsigned x = f()-2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) == 3}}
}

void compare_different_symbol_minus_left_minus_right_int_equal_unsigned() {
  unsigned x = f()-2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) == 1}}
}

void compare_same_symbol_equal_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_plus_left_int_equal_unsigned() {
  unsigned x = f(), y = x;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) + 1U) == (conj_$2{int})}}
}

void compare_same_symbol_minus_left_int_equal_unsigned() {
  unsigned x = f(), y = x;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - 1U) == (conj_$2{int})}}
}

void compare_same_symbol_plus_right_int_equal_unsigned() {
  unsigned x = f(), y = x+1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{(conj_$2{int}) == ((conj_$2{int}) + 1U)}}
}

void compare_same_symbol_minus_right_int_equal_unsigned() {
  unsigned x = f(), y = x-1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{(conj_$2{int}) == ((conj_$2{int}) - 1U)}}
}

void compare_same_symbol_plus_left_plus_right_int_equal_unsigned() {
  unsigned x = f(), y = x+1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_plus_left_minus_right_int_equal_unsigned() {
  unsigned x = f(), y = x-1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) + 1U) == ((conj_$2{int}) - 1U)}}
}

void compare_same_symbol_minus_left_plus_right_int_equal_unsigned() {
  unsigned x = f(), y = x+1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(x == y);
  // expected-warning@-1{{((conj_$2{int}) - 1U) == ((conj_$2{int}) + 1U)}}
}

void compare_same_symbol_minus_left_minus_right_int_equal_unsigned() {
  unsigned x = f(), y = x-1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x == y);
  // expected-warning@-1{{TRUE}}
}

void compare_different_symbol_less_or_equal_unsigned() {
  unsigned x = f(), y = f();
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) <= 0}}
}

void compare_different_symbol_plus_left_int_less_or_equal_unsigned() {
  unsigned x = f()+1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) >= 1}}
}

void compare_different_symbol_minus_left_int_less_or_equal_unsigned() {
  unsigned x = f()-1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) <= 1}}
}

void compare_different_symbol_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = f()+2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 2}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) <= 2}}
}

void compare_different_symbol_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = f()-2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 2}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) >= 2}}
}

void compare_different_symbol_plus_left_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f()+2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) >= 1}}
}

void compare_different_symbol_plus_left_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f()+2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) >= 3}}
}

void compare_different_symbol_minus_left_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f()-2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) <= 3}}
}

void compare_different_symbol_minus_left_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f()-2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) <= 1}}
}

void compare_same_symbol_less_or_equal_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_plus_left_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) + 1U) <= (conj_$2{int})}}
}

void compare_same_symbol_minus_left_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - 1U) <= (conj_$2{int})}}
}

void compare_same_symbol_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x+1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{(conj_$2{int}) <= ((conj_$2{int}) + 1U)}}
}

void compare_same_symbol_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x-1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{(conj_$2{int}) <= ((conj_$2{int}) - 1U)}}
}

void compare_same_symbol_plus_left_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x+1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{TRUE}}
}

void compare_same_symbol_plus_left_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x-1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) + 1U) <= ((conj_$2{int}) - 1U)}}
}

void compare_same_symbol_minus_left_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x+1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(x <= y);
  // expected-warning@-1{{((conj_$2{int}) - 1U) <= ((conj_$2{int}) + 1U)}}
}

void compare_same_symbol_minus_left_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x-1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x <= y);
  // expected-warning@-1{{TRUE}}
}

void compare_different_symbol_less_unsigned() {
  unsigned x = f(), y = f();
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) < 0}}
}

void compare_different_symbol_plus_left_int_less_unsigned() {
  unsigned x = f()+1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) > 1}}
}

void compare_different_symbol_minus_left_int_less_unsigned() {
  unsigned x = f()-1, y = f();
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$9{int}}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) < 1}}
}

void compare_different_symbol_plus_right_int_less_unsigned() {
  unsigned x = f(), y = f()+2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 2}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) < 2}}
}

void compare_different_symbol_minus_right_int_less_unsigned() {
  unsigned x = f(), y = f()-2;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 2}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) > 2}}
}

void compare_different_symbol_plus_left_plus_right_int_less_unsigned() {
  unsigned x = f()+2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) > 1}}
}

void compare_different_symbol_plus_left_minus_right_int_less_unsigned() {
  unsigned x = f()+2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$9{int}) - (conj_$2{int})) > 3}}
}

void compare_different_symbol_minus_left_plus_right_int_less_unsigned() {
  unsigned x = f()-2, y = f()+1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) + 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) < 3}}
}

void compare_different_symbol_minus_left_minus_right_int_less_unsigned() {
  unsigned x = f()-2, y = f()-1;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 2}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$9{int}) - 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - (conj_$9{int})) < 1}}
}

void compare_same_symbol_less_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_plus_left_int_less_unsigned() {
  unsigned x = f(), y = x;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) + 1U) < (conj_$2{int})}}
}

void compare_same_symbol_minus_left_int_less_unsigned() {
  unsigned x = f(), y = x;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - 1U) < (conj_$2{int})}}
}

void compare_same_symbol_plus_right_int_less_unsigned() {
  unsigned x = f(), y = x+1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{(conj_$2{int}) < ((conj_$2{int}) + 1U)}}
}

void compare_same_symbol_minus_right_int_less_unsigned() {
  unsigned x = f(), y = x-1;
  clang_analyzer_dump(x); // expected-warning{{conj_$2{int}}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{(conj_$2{int}) < ((conj_$2{int}) - 1U)}}
}

void compare_same_symbol_plus_left_plus_right_int_less_unsigned() {
  unsigned x = f(), y = x+1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{FALSE}}
}

void compare_same_symbol_plus_left_minus_right_int_less_unsigned() {
  unsigned x = f(), y = x-1;
  ++x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) + 1U) < ((conj_$2{int}) - 1U)}}
}

void compare_same_symbol_minus_left_plus_right_int_less_unsigned() {
  unsigned x = f(), y = x+1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) + 1}}
  clang_analyzer_dump(x < y);
  // expected-warning@-1{{((conj_$2{int}) - 1U) < ((conj_$2{int}) + 1U)}}
}

void compare_same_symbol_minus_left_minus_right_int_less_unsigned() {
  unsigned x = f(), y = x-1;
  --x;
  clang_analyzer_dump(x); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_dump(y); // expected-warning{{(conj_$2{int}) - 1}}
  clang_analyzer_eval(x < y);
  // expected-warning@-1{{FALSE}}
}

void overflow(signed char n, signed char m) {
  if (n + 0 > m + 0) {
    clang_analyzer_eval(n - 126 == m + 3); // expected-warning{{UNKNOWN}}
  }
}

int mixed_integer_types(int x, int y) {
  short a = x - 1U;
  return a - y;
}
