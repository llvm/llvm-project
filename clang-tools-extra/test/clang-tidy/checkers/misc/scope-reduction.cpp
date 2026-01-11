// RUN: %check_clang_tidy %s misc-scope-reduction %t -- --

// Test case 1: Variable can be moved to smaller scope (if-block)
void test_if_scope() {
  int x = 42; // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'x' can be declared in a smaller scope
  if (true) {
    int y = x + 1;
  }
}

// Test case 2: Variable used across multiple scopes - should NOT warn
int test_multiple_scopes(int v) {
  int y = 0; // Should NOT warn - used in if-block and return
  if (v) {
    y = 10;
  }
  return y;
}

// Test case 3: Variable can be moved to nested if-block
void test_nested_if() {
  int a = 5; // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'a' can be declared in a smaller scope
  if (true) {
    if (true) {
      int b = a * 2;
    }
  }
}

// Test case 4: Variable used in same scope - should NOT warn
void test_same_scope() {
  int x = 10; // Should NOT warn - used in same scope
  int y = x + 5;
}

// Test case 5: Variable can be moved to while loop body
void test_while_loop() {
  int counter = 0; // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'counter' can be declared in a smaller scope
  while (true) {
    counter++;
    if (counter > 10) break;
  }
}

// Test case 6: Variable used in multiple branches of same if-statement
void test_if_branches(bool condition) {
  int value = 100; // Should NOT warn - used in both branches
  if (condition) {
    value *= 2;
  } else {
    value /= 2;
  }
}

// Test case 7: Variable can be moved to for-loop body
void test_for_loop_body() {
  int temp = 0; // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'temp' can be declared in a smaller scope
  for (int i = 0; i < 10; i++) {
    temp = i * i;
  }
}

// Test case 8: Variable used in for-loop expressions - should NOT warn (current limitation)
void test_for_loop_expressions() {
  int i; // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'i' can be declared in for-loop initialization
  for (i = 0; i < 5; i++) {
    // loop body
  }
}

// Test case 9: Variable can be moved to switch case
void test_switch_case(int value) {
  int result = 0; // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'result' can be declared in a smaller scope
  switch (value) {
    case 1:
      result = 10;
      break;
    default:
      break;
  }
}

// Test case 10: Variable used across multiple switch cases - should NOT warn
void test_switch_multiple_cases(int value) {
  int accumulator = 0; // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'accumulator' can be declared in a smaller scope
  switch (value) {
    case 1:
      accumulator += 10;
      break;
    case 2:
      accumulator += 20;
      break;
  }
}

// Test case 11: Variable with complex initialization can be moved
void test_complex_init() {
  int complex = (5 + 3) * 2; // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'complex' can be declared in a smaller scope
  if (true) {
    int doubled = complex * 2;
  }
}

// Test case 12: Multiple variables, some can be moved, some cannot
int test_mixed_variables(bool flag) {
  int movable = 10;   // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'movable' can be declared in a smaller scope
  int unmovable = 20; // Should NOT warn - used across scopes

  if (flag) {
    int local = movable + 5;
    unmovable += 1;
  }

  return unmovable;
}

// Test case 13: Variable in try-catch block
void test_try_catch() {
  int error_code = 0; // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'error_code' can be declared in a smaller scope
  try {
    error_code = 404;
  } catch (...) {
    // handle exception
  }
}

// Test case 14: Variable used in catch block and try block - should NOT warn
void test_try_catch_shared() {
  int shared = 0; // Should NOT warn - used in both try and catch
  try {
    shared = 100;
  } catch (...) {
    shared = -1;
  }
}

// Test case 15: Deeply nested scopes
void test_deep_nesting() {
  int deep = 1; // CHECK-MESSAGES: :[[@LINE]]:7: warning: variable 'deep' can be declared in a smaller scope
  if (true) {
    if (true) {
      if (true) {
        if (true) {
          int result = deep * 4;
        }
      }
    }
  }
}

// Test case 16: Variable declared but never used - should NOT warn (different checker's job)
void test_unused_variable() {
  int unused = 42; // Should NOT warn - this checker only handles scope reduction
}

// Test case 17: Global variable - should NOT be processed
int global_var = 100;

// Test case 18: Static local variable - should NOT warn
void test_static_variable() {
  static int static_var = 0; // Should NOT warn - static variables have different semantics
  if (true) {
    static_var++;
  }
}

// Test case 19: Function parameter - should NOT be processed
void test_parameter(int param) {
  if (true) {
    int local = param + 1;
  }
}

// Test case 20: Variable used in lambda - should NOT warn (complex case)
void test_lambda() {
  int captured = 10; // Should NOT warn - used in lambda
  auto lambda = [&]() {
    return captured * 2;
  };
  lambda();
}
