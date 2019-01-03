// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection,core.builtin -analyzer-config aggressive-binary-operation-simplification=true -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(int x);
void clang_analyzer_denote(int x, const char *literal);
void clang_analyzer_express(int x);

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
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(x == y); // expected-warning {{$x - $y == 0}}
}

void compare_different_symbol_plus_left_int_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x += 1;
  clang_analyzer_express(x == y); // expected-warning {{$y - $x == 1}}
}

void compare_different_symbol_minus_left_int_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x -= 1;
  clang_analyzer_express(x == y); // expected-warning {{$x - $y == 1}}
}

void compare_different_symbol_plus_right_int_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  y += 2;
  clang_analyzer_express(y); // expected-warning {{$y + 2}}
  clang_analyzer_express(x == y); // expected-warning {{$x - $y == 2}}
}

void compare_different_symbol_minus_right_int_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  y -= 2;
  clang_analyzer_express(y); // expected-warning {{$y - 2}}
  clang_analyzer_express(x == y); // expected-warning {{$y - $x == 2}}
}

void compare_different_symbol_plus_left_plus_right_int_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x += 2;
  y += 1;
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x == y); // expected-warning {{$y - $x == 1}}
}

void compare_different_symbol_plus_left_minus_right_int_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x += 2;
  y -= 1;
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x == y); // expected-warning {{$y - $x == 3}}
}

void compare_different_symbol_minus_left_plus_right_int_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x -= 2;
  y += 1;
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x == y); // expected-warning {{$x - $y == 3}}
}

void compare_different_symbol_minus_left_minus_right_int_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x -= 2;
  y -= 1;
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x == y); // expected-warning {{$x - $y == 1}}
}

void compare_same_symbol_equal() {
  int x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x == y); // expected-warning {{TRUE}}
}

void compare_same_symbol_plus_left_int_equal() {
  int x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x == y); // expected-warning {{FALSE}}
}

void compare_same_symbol_minus_left_int_equal() {
  int x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x == y); // expected-warning {{FALSE}}
}

void compare_same_symbol_plus_right_int_equal() {
  int x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x == y); // expected-warning {{FALSE}}
}

void compare_same_symbol_minus_right_int_equal() {
  int x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x == y); // expected-warning {{FALSE}}
}

void compare_same_symbol_plus_left_plus_right_int_equal() {
  int x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x == y); // expected-warning {{TRUE}}
}

void compare_same_symbol_plus_left_minus_right_int_equal() {
  int x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x == y); // expected-warning {{FALSE}}
}

void compare_same_symbol_minus_left_plus_right_int_equal() {
  int x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x == y); // expected-warning {{FALSE}}
}

void compare_same_symbol_minus_left_minus_right_int_equal() {
  int x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x == y); // expected-warning {{TRUE}}
}

void compare_different_symbol_less_or_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(x <= y); // expected-warning {{$x - $y <= 0}}
}

void compare_different_symbol_plus_left_int_less_or_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x += 1;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$y - $x >= 1}}
}

void compare_different_symbol_minus_left_int_less_or_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x -= 1;
  clang_analyzer_express(x <= y); // expected-warning {{$x - $y <= 1}}
}

void compare_different_symbol_plus_right_int_less_or_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  y += 2;
  clang_analyzer_express(x <= y); // expected-warning {{$x - $y <= 2}}
}

void compare_different_symbol_minus_right_int_less_or_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  y -= 2;
  clang_analyzer_express(y); // expected-warning {{$y - 2}}
  clang_analyzer_express(x <= y); // expected-warning {{$y - $x >= 2}}
}

void compare_different_symbol_plus_left_plus_right_int_less_or_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x += 2;
  y += 1;
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$y - $x >= 1}}
}

void compare_different_symbol_plus_left_minus_right_int_less_or_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x += 2;
  y -= 1;
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$y - $x >= 3}}
}

void compare_different_symbol_minus_left_plus_right_int_less_or_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x -= 2;
  y += 1;
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$x - $y <= 3}}
}

void compare_different_symbol_minus_left_minus_right_int_less_or_equal() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x -= 2;
  y -= 1;
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$x - $y <= 1}}
}

void compare_same_symbol_less_or_equal() {
  int x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x <= y); // expected-warning {{TRUE}}
}

void compare_same_symbol_plus_left_int_less_or_equal() {
  int x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x <= y); // expected-warning {{FALSE}}
}

void compare_same_symbol_minus_left_int_less_or_equal() {
  int x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x <= y); // expected-warning {{TRUE}}
}

void compare_same_symbol_plus_right_int_less_or_equal() {
  int x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x <= y); // expected-warning {{TRUE}}
}

void compare_same_symbol_minus_right_int_less_or_equal() {
  int x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x <= y); // expected-warning {{FALSE}}
}

void compare_same_symbol_plus_left_plus_right_int_less_or_equal() {
  int x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x <= y); // expected-warning {{TRUE}}
}

void compare_same_symbol_plus_left_minus_right_int_less_or_equal() {
  int x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x <= y); // expected-warning {{FALSE}}
}

void compare_same_symbol_minus_left_plus_right_int_less_or_equal() {
  int x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x <= y); // expected-warning {{TRUE}}
}

void compare_same_symbol_minus_left_minus_right_int_less_or_equal() {
  int x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x <= y); // expected-warning {{TRUE}}
}

void compare_different_symbol_less() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x < y); // expected-warning {{$x - $y < 0}}
}

void compare_different_symbol_plus_left_int_less() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x += 1;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x < y); // expected-warning {{$y - $x > 1}}
}

void compare_different_symbol_minus_left_int_less() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x -= 1;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x < y); // expected-warning {{$x - $y < 1}}
}

void compare_different_symbol_plus_right_int_less() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  y += 2;
  clang_analyzer_express(y); // expected-warning {{$y + 2}}
  clang_analyzer_express(x < y); // expected-warning {{$x - $y < 2}}
}

void compare_different_symbol_minus_right_int_less() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  y -= 2;
  clang_analyzer_express(y); // expected-warning {{$y - 2}}
  clang_analyzer_express(x < y); // expected-warning {{$y - $x > 2}}
}

void compare_different_symbol_plus_left_plus_right_int_less() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x += 2;
  y += 1;
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x < y); // expected-warning {{$y - $x > 1}}
}

void compare_different_symbol_plus_left_minus_right_int_less() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x += 2;
  y -= 1;
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x < y); // expected-warning {{$y - $x > 3}}
}

void compare_different_symbol_minus_left_plus_right_int_less() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x -= 2;
  y += 1;
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x < y); // expected-warning {{$x - $y < 3}}
}

void compare_different_symbol_minus_left_minus_right_int_less() {
  int x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  x -= 2;
  y -= 1;
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x < y); // expected-warning {{$x - $y < 1}}
}

void compare_same_symbol_less() {
  int x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x < y); // expected-warning {{FALSE}}
}

void compare_same_symbol_plus_left_int_less() {
  int x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x < y); // expected-warning {{FALSE}}
}

void compare_same_symbol_minus_left_int_less() {
  int x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x < y); // expected-warning {{TRUE}}
}

void compare_same_symbol_plus_right_int_less() {
  int x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x < y); // expected-warning {{TRUE}}
}

void compare_same_symbol_minus_right_int_less() {
  int x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x < y); // expected-warning {{FALSE}}
}

void compare_same_symbol_plus_left_plus_right_int_less() {
  int x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x < y); // expected-warning {{FALSE}}
}

void compare_same_symbol_plus_left_minus_right_int_less() {
  int x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x < y); // expected-warning {{FALSE}}
}

void compare_same_symbol_minus_left_plus_right_int_less() {
  int x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x < y); // expected-warning {{TRUE}}
}

void compare_same_symbol_minus_left_minus_right_int_less() {
  int x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x < y); // expected-warning {{FALSE}}
}

void compare_different_symbol_equal_unsigned() {
  unsigned x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x == y); // expected-warning {{$x - $y == 0}}
}

void compare_different_symbol_plus_left_int_equal_unsigned() {
  unsigned x = f() + 1, y = f();
  clang_analyzer_denote(x - 1, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x == y); // expected-warning {{$y - $x == 1}}
}

void compare_different_symbol_minus_left_int_equal_unsigned() {
  unsigned x = f() - 1, y = f();
  clang_analyzer_denote(x + 1, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x == y); // expected-warning {{$x - $y == 1}}
}

void compare_different_symbol_plus_right_int_equal_unsigned() {
  unsigned x = f(), y = f() + 2;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y - 2, "$y");
  clang_analyzer_express(y); // expected-warning {{$y + 2}}
  clang_analyzer_express(x == y); // expected-warning {{$x - $y == 2}}
}

void compare_different_symbol_minus_right_int_equal_unsigned() {
  unsigned x = f(), y = f() - 2;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y + 2, "$y");
  clang_analyzer_express(y); // expected-warning {{$y - 2}}
  clang_analyzer_express(x == y); // expected-warning {{$y - $x == 2}}
}

void compare_different_symbol_plus_left_plus_right_int_equal_unsigned() {
  unsigned x = f() + 2, y = f() + 1;
  clang_analyzer_denote(x - 2, "$x");
  clang_analyzer_denote(y - 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x == y); // expected-warning {{$y - $x == 1}}
}

void compare_different_symbol_plus_left_minus_right_int_equal_unsigned() {
  unsigned x = f() + 2, y = f() - 1;
  clang_analyzer_denote(x - 2, "$x");
  clang_analyzer_denote(y + 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x == y); // expected-warning {{$y - $x == 3}}
}

void compare_different_symbol_minus_left_plus_right_int_equal_unsigned() {
  unsigned x = f() - 2, y = f() + 1;
  clang_analyzer_denote(x + 2, "$x");
  clang_analyzer_denote(y - 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x == y); // expected-warning {{$x - $y == 3}}
}

void compare_different_symbol_minus_left_minus_right_int_equal_unsigned() {
  unsigned x = f() - 2, y = f() - 1;
  clang_analyzer_denote(x + 2, "$x");
  clang_analyzer_denote(y + 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x == y); // expected-warning {{$x - $y == 1}}
}

void compare_same_symbol_equal_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x == y); // expected-warning {{TRUE}}
}

void compare_same_symbol_plus_left_int_equal_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_express(x == y); // expected-warning {{$x + 1U == $x}}
}

void compare_same_symbol_minus_left_int_equal_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_express(x == y); // expected-warning {{$x - 1U == $x}}
}

void compare_same_symbol_plus_right_int_equal_unsigned() {
  unsigned x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_express(x == y); // expected-warning {{$x == $x + 1U}}
}

void compare_same_symbol_minus_right_int_equal_unsigned() {
  unsigned x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_express(x == y); // expected-warning {{$x == $x - 1U}}
}

void compare_same_symbol_plus_left_plus_right_int_equal_unsigned() {
  unsigned x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x == y); // expected-warning {{TRUE}}
}

void compare_same_symbol_plus_left_minus_right_int_equal_unsigned() {
  unsigned x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_express(x == y); // expected-warning {{$x + 1U == $x - 1U}}
}

void compare_same_symbol_minus_left_plus_right_int_equal_unsigned() {
  unsigned x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_express(x == y); // expected-warning {{$x - 1U == $x + 1U}}
}

void compare_same_symbol_minus_left_minus_right_int_equal_unsigned() {
  unsigned x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x == y); // expected-warning {{TRUE}}
}

void compare_different_symbol_less_or_equal_unsigned() {
  unsigned x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x <= y); // expected-warning {{$x - $y <= 0}}
}

void compare_different_symbol_plus_left_int_less_or_equal_unsigned() {
  unsigned x = f() + 1, y = f();
  clang_analyzer_denote(x - 1, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x <= y); // expected-warning {{$y - $x >= 1}}
}

void compare_different_symbol_minus_left_int_less_or_equal_unsigned() {
  unsigned x = f() - 1, y = f();
  clang_analyzer_denote(x + 1, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x <= y); // expected-warning {{$x - $y <= 1}}
}

void compare_different_symbol_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = f() + 2;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y - 2, "$y");
  clang_analyzer_express(y); // expected-warning {{$y + 2}}
  clang_analyzer_express(x <= y); // expected-warning {{$x - $y <= 2}}
}

void compare_different_symbol_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = f() - 2;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y + 2, "$y");
  clang_analyzer_express(y); // expected-warning {{$y - 2}}
  clang_analyzer_express(x <= y); // expected-warning {{$y - $x >= 2}}
}

void compare_different_symbol_plus_left_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f() + 2, y = f() + 1;
  clang_analyzer_denote(x - 2, "$x");
  clang_analyzer_denote(y - 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$y - $x >= 1}}
}

void compare_different_symbol_plus_left_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f() + 2, y = f() - 1;
  clang_analyzer_denote(x - 2, "$x");
  clang_analyzer_denote(y + 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$y - $x >= 3}}
}

void compare_different_symbol_minus_left_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f() - 2, y = f() + 1;
  clang_analyzer_denote(x + 2, "$x");
  clang_analyzer_denote(y - 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$x - $y <= 3}}
}

void compare_different_symbol_minus_left_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f() - 2, y = f() - 1;
  clang_analyzer_denote(x + 2, "$x");
  clang_analyzer_denote(y + 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$x - $y <= 1}}
}

void compare_same_symbol_less_or_equal_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x <= y); // expected-warning {{TRUE}}
}

void compare_same_symbol_plus_left_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_express(x <= y); // expected-warning {{$x + 1U <= $x}}
}

void compare_same_symbol_minus_left_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_express(x <= y); // expected-warning {{$x - 1U <= $x}}
}

void compare_same_symbol_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$x <= $x + 1U}}
}

void compare_same_symbol_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$x <= $x - 1U}}
}

void compare_same_symbol_plus_left_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x <= y); // expected-warning {{TRUE}}
}

void compare_same_symbol_plus_left_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$x + 1U <= $x - 1U}}
}

void compare_same_symbol_minus_left_plus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_express(x <= y); // expected-warning {{$x - 1U <= $x + 1U}}
}

void compare_same_symbol_minus_left_minus_right_int_less_or_equal_unsigned() {
  unsigned x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x <= y); // expected-warning {{TRUE}}
}

void compare_different_symbol_less_unsigned() {
  unsigned x = f(), y = f();
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x < y); // expected-warning {{$x - $y < 0}}
}

void compare_different_symbol_plus_left_int_less_unsigned() {
  unsigned x = f() + 1, y = f();
  clang_analyzer_denote(x - 1, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x < y); // expected-warning {{$y - $x > 1}}
}

void compare_different_symbol_minus_left_int_less_unsigned() {
  unsigned x = f() - 1, y = f();
  clang_analyzer_denote(x + 1, "$x");
  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x < y); // expected-warning {{$x - $y < 1}}
}

void compare_different_symbol_plus_right_int_less_unsigned() {
  unsigned x = f(), y = f() + 2;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y - 2, "$y");
  clang_analyzer_express(y); // expected-warning {{$y + 2}}
  clang_analyzer_express(x < y); // expected-warning {{$x - $y < 2}}
}

void compare_different_symbol_minus_right_int_less_unsigned() {
  unsigned x = f(), y = f() - 2;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_denote(y + 2, "$y");
  clang_analyzer_express(y); // expected-warning {{$y - 2}}
  clang_analyzer_express(x < y); // expected-warning {{$y - $x > 2}}
}

void compare_different_symbol_plus_left_plus_right_int_less_unsigned() {
  unsigned x = f() + 2, y = f() + 1;
  clang_analyzer_denote(x - 2, "$x");
  clang_analyzer_denote(y - 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x < y); // expected-warning {{$y - $x > 1}}
}

void compare_different_symbol_plus_left_minus_right_int_less_unsigned() {
  unsigned x = f() + 2, y = f() - 1;
  clang_analyzer_denote(x - 2, "$x");
  clang_analyzer_denote(y + 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x + 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x < y); // expected-warning {{$y - $x > 3}}
}

void compare_different_symbol_minus_left_plus_right_int_less_unsigned() {
  unsigned x = f() - 2, y = f() + 1;
  clang_analyzer_denote(x + 2, "$x");
  clang_analyzer_denote(y - 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y + 1}}
  clang_analyzer_express(x < y); // expected-warning {{$x - $y < 3}}
}

void compare_different_symbol_minus_left_minus_right_int_less_unsigned() {
  unsigned x = f() - 2, y = f() - 1;
  clang_analyzer_denote(x + 2, "$x");
  clang_analyzer_denote(y + 1, "$y");
  clang_analyzer_express(x); // expected-warning {{$x - 2}}
  clang_analyzer_express(y); // expected-warning {{$y - 1}}
  clang_analyzer_express(x < y); // expected-warning {{$x - $y < 1}}
}

void compare_same_symbol_less_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_eval(x < y); // expected-warning {{FALSE}}
}

void compare_same_symbol_plus_left_int_less_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_express(x < y); // expected-warning {{$x + 1U < $x}}
}

void compare_same_symbol_minus_left_int_less_unsigned() {
  unsigned x = f(), y = x;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x}}
  clang_analyzer_express(x < y); // expected-warning {{$x - 1U < $x}}
}

void compare_same_symbol_plus_right_int_less_unsigned() {
  unsigned x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_express(x < y); // expected-warning {{$x < $x + 1U}}
}

void compare_same_symbol_minus_right_int_less_unsigned() {
  unsigned x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_express(x < y); // expected-warning {{$x < $x - 1U}}
}

void compare_same_symbol_plus_left_plus_right_int_less_unsigned() {
  unsigned x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_eval(x < y); // expected-warning {{FALSE}}
}

void compare_same_symbol_plus_left_minus_right_int_less_unsigned() {
  unsigned x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  ++x;
  clang_analyzer_express(x); // expected-warning {{$x + 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_express(x < y); // expected-warning {{$x + 1U < $x - 1U}}
}

void compare_same_symbol_minus_left_plus_right_int_less_unsigned() {
  unsigned x = f(), y = x + 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x + 1}}
  clang_analyzer_express(x < y); // expected-warning {{$x - 1U < $x + 1U}}
}

void compare_same_symbol_minus_left_minus_right_int_less_unsigned() {
  unsigned x = f(), y = x - 1;
  clang_analyzer_denote(x, "$x");
  --x;
  clang_analyzer_express(x); // expected-warning {{$x - 1}}
  clang_analyzer_express(y); // expected-warning {{$x - 1}}
  clang_analyzer_eval(x < y); // expected-warning {{FALSE}}
}

void overflow(signed char n, signed char m) {
  if (n + 0 > m + 0) {
    clang_analyzer_eval(n - 126 == m + 3); // expected-warning {{UNKNOWN}}
  }
}

int mixed_integer_types(int x, int y) {
  short a = x - 1U;
  return a - y;
}

unsigned gu();
unsigned fu() {
  unsigned x = gu();
  // Assert that no overflows occur in this test file.
  // Assuming that concrete integers are also within that range.
  assert(x <= ((unsigned)UINT_MAX / 4));
  return x;
}

void unsigned_concrete_int_no_crash() {
  unsigned x = fu() + 1U, y = fu() + 1U;
  clang_analyzer_denote(x - 1U, "$x");
  clang_analyzer_denote(y - 1U, "$y");
  clang_analyzer_express(y); // expected-warning {{$y}}
  clang_analyzer_express(x == y); // expected-warning {{$x + 1U == $y + 1U}}
}
