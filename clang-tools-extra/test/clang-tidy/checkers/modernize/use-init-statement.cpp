// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-init-statement %t

int compute();

// Positive: simple int variable used in if condition
void test_if_simple_int() {
  int x = 42;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: variable 'x' can be declared in the 'if' init-statement [modernize-use-init-statement]
  // CHECK-FIXES: if (int x = 42; x > 0) {
  if (x > 0) {
    int y = x + 1;
  }
}

// Positive: function call result used in if condition
void test_if_function_call() {
  int result = compute();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: variable 'result' can be declared in the 'if' init-statement [modernize-use-init-statement]
  // CHECK-FIXES: if (int result = compute(); result != 0) {
  if (result != 0) {
    int y = result;
  }
}

// Positive: variable only used in condition, not in body
void test_if_only_in_condition() {
  int x = 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: variable 'x' can be declared in the 'if' init-statement [modernize-use-init-statement]
  // CHECK-FIXES: if (int x = 10; x > 5) {
  if (x > 5) {
  }
}

enum Color { Red, Green, Blue };
Color getColor();

// Positive: switch statement
void test_switch() {
  Color c = getColor();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: variable 'c' can be declared in the 'switch' init-statement [modernize-use-init-statement]
  // CHECK-FIXES: switch (Color c = getColor(); c) {
  switch (c) {
  case Red:
    break;
  case Green:
    break;
  default:
    break;
  }
}

// Negative: variable used after if
void test_used_after_if() {
  int it = compute();
  if (it != 0) {
    int y = it;
  }
  int z = it; // Used after if - should not move
}

// Negative: variable modified after if
void test_used_in_else_and_after() {
  int x = compute();
  if (x > 0) {
    int y = x;
  }
  x = 0; // Modified after if
}

// Negative: no initializer
void test_no_init() {
  int x;
  if (x > 0) {
  }
}

// Negative: already has init-statement
void test_already_has_init() {
  if (int x = compute(); x > 0) {
  }
}

// Negative: multiple declarations
void test_multiple_decls() {
  int x = 1, y = 2;
  if (x > 0) {
  }
}

// Negative: variable not referenced in condition
void test_not_in_condition() {
  int x = 10;
  if (true) {
    int y = x;
  }
}

// Negative: static variable
void test_static_var() {
  static int x = 42;
  if (x > 0) {
  }
}

// Negative: previous stmt is not the variable declaration
void test_not_previous_stmt() {
  int x = 10;
  int y = 20;
  if (x > 0) {
  }
}
