// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-constexpr %t

// Positive: integer literal
void test_int_literal() {
  const int x = 42;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'x' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES: constexpr int x = 42;
}

// Positive: arithmetic expression of literals
void test_arithmetic() {
  const int y = 2 + 3 * 4;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'y' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES: constexpr int y = 2 + 3 * 4;
}

// Positive: bool literal
void test_bool() {
  const bool b = true;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: variable 'b' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES: constexpr bool b = true;
}

// Positive: char literal
void test_char() {
  const char c = 'a';
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: variable 'c' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES: constexpr char c = 'a';
}

// Positive: float literal
void test_float() {
  const double d = 3.14;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: variable 'd' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES: constexpr double d = 3.14;
}

// Positive: enum value
enum Color { Red, Green, Blue };
void test_enum() {
  const Color c = Red;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: variable 'c' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES: constexpr Color c = Red;
}

// Positive: sizeof expression
void test_sizeof() {
  const int s = sizeof(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 's' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES: constexpr int s = sizeof(int);
}

// Positive: east const style
void test_east_const() {
  int const x = 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'x' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES: int constexpr x = 10;
}

// Negative: already constexpr
void test_already_constexpr() {
  constexpr int x = 42;
}

// Negative: non-const variable
void test_non_const() {
  int x = 42;
}

// Negative: non-constant initializer
int compute();
void test_non_constant_init() {
  const int x = compute();
}

// Negative: reference type
void test_reference() {
  int val = 42;
  const int &r = val;
}

// Negative: volatile
void test_volatile() {
  const volatile int x = 42;
}

// Negative: static local
void test_static() {
  static const int x = 42;
}

// Negative: non-literal type (std::string mock)
struct NonLiteral {
  NonLiteral(const char *);
  ~NonLiteral();
};
void test_non_literal() {
  const NonLiteral s = "hello";
}

// Negative: pointer type (fix-it is non-trivial)
void test_pointer() {
  int *const p = nullptr;
  const int *q = nullptr;
}

// Negative: global variable (not local)
const int global_val = 42;

// Negative: variable depends on function argument
void test_depends_on_arg(int n) {
  const int x = n + 1;
}
