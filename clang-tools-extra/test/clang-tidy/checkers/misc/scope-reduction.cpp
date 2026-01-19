// RUN: %check_clang_tidy %s misc-scope-reduction %t -- --

// Variable can be moved to smaller scope (if-block)
void test_if_scope() {
  int x = 42;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'x' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:13: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int y = x + 1;
  }
}

// Variable used across multiple scopes - should NOT warn
int test_multiple_scopes(int v) {
  int y = 0; // Should NOT warn - used in if-block and return
  if (v) {
    y = 10;
  }
  return y;
}

// Variable can be moved to nested if-block
void test_nested_if() {
  int a = 5;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'a' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:15: note: used here
  // CHECK-NOTES: :[[@LINE+2]]:15: note: can be declared in this scope
  if (true) {
    if (true) {
      int b = a * 2;
    }
  }
}

// Variable used in same scope - should NOT warn
void test_same_scope() {
  int x = 10; // Should NOT warn - used in same scope
  int y = x + 5;
}

// Variable can be moved to while loop body
// FIXME: This is a false positive. Correcting this will require
//        loop semantic comprehension and var lifetime analysis.
void test_while_loop() {
  int counter = 0;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'counter' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+4]]:9: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:16: note: can be declared in this scope
  while (true) {
    counter++;
    if (counter > 10) break;
  }
}

// Variable used in multiple branches of same if-statement
void test_if_branches(bool condition) {
  int value = 100; // Should NOT warn - used in both branches
  if (condition) {
    value *= 2;
  } else {
    value /= 2;
  }
}

// Variable can be moved to for-loop body
void test_for_loop_body() {
  int temp = 0;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'temp' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:32: note: can be declared in this scope
  for (int i = 0; i < 10; i++) {
    temp = i * i;
  }
}

// Variable used in for-loop expressions
void test_for_loop_expressions() {
  int i;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'i' can be declared in for-loop initialization
  // CHECK-NOTES: :[[@LINE+1]]:3: note: can be declared in this for-loop
  for (i = 0; i < 5; i++) {
    // loop body
  }
}

// Variable can be moved to switch case
void test_switch_case(int value) {
  int result = 0;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'result' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:7: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:18: note: can be declared in this scope
  switch (value) {
    case 1:
      result = 10;
      break;
    default:
      break;
  }
}

// Variable used across multiple switch cases - should NOT warn
void test_switch_multiple_cases(int value) {
  int accumulator = 0;
  switch (value) {
    case 1:
      accumulator += 10;
      break;
    case 2:
      accumulator += 20;
      break;
  }
}

// Variable with complex initialization can be moved
void test_complex_init() {
  int cmplx_expr = (5 + 3) * 2;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'cmplx_expr' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:19: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int doubled = cmplx_expr * 2;
  }
}

// Multiple variables, some can be moved, some cannot
int test_mixed_variables(bool flag) {
  int movable = 10;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'movable' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+5]]:17: note: used here
  // CHECK-NOTES: :[[@LINE+3]]:13: note: can be declared in this scope
  int unmovable = 20; // Should NOT warn - used across scopes

  if (flag) {
    int local = movable + 5;
    unmovable += 1;
  }

  return unmovable;
}

// Variable in try-catch block
void test_try_catch() {
  int error_code = 0;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'error_code' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:7: note: can be declared in this scope
  try {
    error_code = 404;
  } catch (...) {
    // handle exception
  }
}

// Variable used in catch block and try block - should NOT warn
void test_try_catch_shared() {
  int shared = 0; // Should NOT warn - used in both try and catch
  try {
    shared = 100;
  } catch (...) {
    shared = -1;
  }
}

// Deeply nested scopes
void test_deep_nesting() {
  int deep = 1;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'deep' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+6]]:24: note: used here
  // CHECK-NOTES: :[[@LINE+4]]:19: note: can be declared in this scope
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

// Variable declared but never used - should NOT warn (different checker's job)
void test_unused_variable() {
  int unused = 42; // Should NOT warn - this checker only handles scope reduction
}

// Global variable - should NOT be processed
int global_var = 100;

namespace GlobalTestNamespace {
  int namespaced_global = 200;

  // Function using global variables - should NOT warn
  void test_global_usage() {
    int local = global_var + namespaced_global;
    // CHECK-NOTES: :[[@LINE-1]]:9: warning: variable 'local' can be declared in a smaller scope
    // CHECK-NOTES: :[[@LINE+3]]:7: note: used here
    // CHECK-NOTES: :[[@LINE+1]]:15: note: can be declared in this scope
    if (true) {
      local *= 2;
    }
  }

  // Global vars used in smaller scopes. Should NOT be detected.
  void test_globals_not_detected() {
    if (true) {
      global_var = 300;
      namespaced_global = 400;
      int result = global_var + namespaced_global;
    }
  }
}

// Static local variable - should NOT warn
void test_static_variable() {
  static int static_var = 0; // Should NOT warn - static variables have different semantics
  if (true) {
    static_var++;
  }
}

// Function parameter - should NOT be processed
void test_parameter(int param) {
  if (true) {
    int local = param + 1;
  }
}

// Variable used in lambda - should NOT warn (complex case)
void test_lambda() {
  int captured = 10; // Should NOT warn - used in lambda
  auto lambda = [&]() {
    return captured * 2;
  };
  lambda();
}

// Variable set from function call, used in if clause
// Should NOT warn. Don't know if func() has side effects
int func();
void test_function_call() {
  int i = func();
  if (true) {
    i = 0;
  }
}

// Variable used inside a loop.
// Should NOT warn.
void test_for_loop_reuse() {
  int temp = 0;
  for (int i = 0; i<10; i++) {
    temp += i;
  }
}

// Variable can be moved closer to lambda usage
void test_lambda_movable() {
  int local = 5;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'local' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+5]]:20: note: used here
  // CHECK-NOTES: :[[@LINE+5]]:14: note: used here
  // CHECK-NOTES: :[[@LINE+2]]:13: note: can be declared in this scope

  if (true) {
    auto lambda = [local]() {
      return local *3;
    };
  }
}

// Variable declared but never used with empty scope after
void test_unused_empty_scope() {
  int unused = 42; // Should NOT warn - this checker only handles scope reduction
  if (true) {
    // empty scope, variable not used here
  }
}

// Variable used in switch and other scope - should warn if common scope allows
void test_switch_mixed_usage(int value) {
  int mixed = 0;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'mixed' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+6]]:9: note: used here
  // CHECK-NOTES: :[[@LINE+8]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    switch (value) {
      case 1:
        mixed = 10;
        break;
    }
    mixed += 5; // Also used outside switch
  }
}

// Variable in nested switch - should warn for single case
void test_nested_switch(int outer, int inner) {
  int nested = 0;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'nested' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+6]]:11: note: used here
  // CHECK-NOTES: :[[@LINE+3]]:22: note: can be declared in this scope
  switch (outer) {
    case 1:
      switch (inner) {
        case 1:
          nested = 42;
          break;
      }
      break;
  }
}

// Variable used in switch default only - should warn
void test_switch_default_only(int value) {
  int def = 0;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'def' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+6]]:7: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:18: note: can be declared in this scope
  switch (value) {
    case 1:
      break;
    default:
      def = 100;
      break;
  }
}

// Variable used in multiple switches - should NOT warn
void test_multiple_switches(int v1, int v2) {
  int multi = 0; // Should NOT warn - used across different switches
  switch (v1) {
    case 1:
      multi = 10;
      break;
  }
  switch (v2) {
    case 1:
      multi = 20;
      break;
  }
}

// Range-based for loop declared variable - should NOT warn
void test_range_for_declared() {
  int vec[] = {1, 2, 3};
  for (auto item : vec) {
    // use item
  }
}

// Variable used in range-based for loop - should NOT warn
void test_range_for_usage() {
  int sum = 0;
  if (true) {
    int vec[] = {1, 2, 3};
    for (auto item : vec) {
      sum += item;
    }
  }
}

// Many variable uses - test diagnostic note limiting
void test_diagnostic_limiting() {
  int x = 42;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'x' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+6]]:13: note: used here
  // CHECK-NOTES: :[[@LINE+6]]:13: note: used here
  // CHECK-NOTES: :[[@LINE+6]]:13: note: used here
  // CHECK-NOTES: :[[@LINE+6]]:13: note: and 3 more uses...
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int a = x + 1;  // First use
    int b = x + 2;  // Second use
    int c = x + 3;  // Third use
    int d = x + 4;  // Fourth use - should show in overflow note
    int e = x + 5;  // Fifth use
    int f = x + 6;  // Sixth use
  }
}

// Exactly 3 uses - no overflow message
void test_exactly_three_uses() {
  int x = 1;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'x' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+5]]:13: note: used here
  // CHECK-NOTES: :[[@LINE+5]]:13: note: used here
  // CHECK-NOTES: :[[@LINE+5]]:13: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int a = x + 1;  // First use
    int b = x + 2;  // Second use
    int c = x + 3;  // Third use
  }
}

// Fewer than 3 uses - show all
void test_few_uses() {
  int x = 1;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'x' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:13: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int a = x + 1;  // First use
  }
}

// For-loop case with many uses - test limiting for for-loop diagnostics
void test_for_loop_limiting() {
  int i;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'i' can be declared in for-loop initialization
  // CHECK-NOTES: :[[@LINE+1]]:3: note: can be declared in this for-loop
  for (i = 0; i < 5; i++) {
    int temp = i; // Fourth use of i
  }
}

// Test case for variables within the for-loop scope. (should NOT be reported)
void testForLoopCase() {
  for (int i = 0; i < 10; ++i) {
    int byte = 0;  // Declared in for-loop scope, used in smaller loop body scope
    byte = i * 2;  // Should NOT be reported - usage scope is smaller
    byte += 1;
  }
}

// Test case for variables used in broader scopes (SHOULD be reported)
void testBroaderScope() {
  int value = 0;  // Should be reported - used in broader if-statement scope
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'value' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+4]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    value = 42;
    value += 1;
  }
}
// Test cases for accumulator pattern detection

// Positive cases - should still warn (compound assignments outside loops)

// Compound assignment outside loop - should warn
void test_compound_assignment_no_loop() {
  int value = 10;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'value' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    value *= 2;  // Not in loop, should still warn
  }
}

// Self-referencing assignment outside loop - should warn
void test_self_reference_no_loop() {
  int x = 5;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'x' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+3]]:9: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    x = x + 10;  // Not in loop, should still warn
  }
}

// Binary operation self-reference outside loop - should warn
void test_binary_self_reference_no_loop() {
  int result = 0;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'result' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+3]]:14: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    result = result + 42;  // Not in loop, should still warn
  }
}

// Negative cases - should NOT warn (accumulator patterns in loops)

// Compound assignment in while loop - should NOT warn
void test_compound_assignment_while() {
  int sum = 0;  // Should NOT warn - accumulator in while loop
  while (sum < 100) {
    sum += 10;
  }
}

// Compound assignment in for loop - should NOT warn
void test_compound_assignment_for() {
  int total = 0;  // Should NOT warn - accumulator in for loop
  for (int i = 0; i < 10; ++i) {
    total += i;
  }
}

// Self-referencing in for loop - should NOT warn
void test_self_reference_for() {
  bool found = false;  // Should NOT warn - accumulator pattern
  for (int i = 0; i < 10; ++i) {
    found = found || (i > 5);
  }
}

// Binary operation self-reference in loop - should NOT warn
void test_binary_self_reference_for() {
  int product = 1;  // Should NOT warn - accumulator pattern
  for (int i = 1; i <= 5; ++i) {
    product = product * i;
  }
}

// Multiple accumulator operations - should NOT warn
void test_multiple_accumulator() {
  int count = 0;  // Should NOT warn
  for (int i = 0; i < 10; ++i) {
    count += i;
    count *= 2;  // Multiple compound assignments in same loop
  }
}

// Range-based for loop accumulator - should NOT warn
void test_range_for_accumulator() {
  int sum = 0;  // Should NOT warn - accumulator in range-based for
  int arr[] = {1, 2, 3, 4, 5};
  for (auto item : arr) {
    sum += item;
  }
}

// Do-while loop accumulator - should NOT warn
void test_do_while_accumulator() {
  int counter = 0;  // Should NOT warn - accumulator in do-while
  do {
    counter++;
  } while (counter < 5);
}

// Edge cases

// Nested loops with accumulator - should NOT warn
void test_nested_loop_accumulator() {
  int total = 0;  // Should NOT warn
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      total += i * j;  // Accumulator in nested loop
    }
  }
}

// Accumulator in inner loop only - should NOT warn
void test_inner_loop_accumulator() {
  int sum = 0;  // Should NOT warn - used as accumulator in inner loop
  if (true) {
    for (int i = 0; i < 10; ++i) {
      sum += i;
    }
  }
}

// Mixed usage - accumulator + other usage - complex case
void test_mixed_accumulator_usage() {
  int value = 0;  // Complex case - used as accumulator AND in other scope
  for (int i = 0; i < 5; ++i) {
    value += i;  // Accumulator usage
  }
  if (true) {
    value = 100;  // Non-accumulator usage - this makes it complex
  }
}

// Variable used in loop but not as accumulator - should warn
void test_non_accumulator_in_loop() {
  int temp = 42;  // Used in loop but not modified - should warn
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'temp' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:18: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:32: note: can be declared in this scope
  for (int i = 0; i < 10; ++i) {
    int result = temp * 2;  // Just reading temp, not modifying it
  }
}

// Compound assignment with different variable - should warn
void test_compound_different_var() {
  int x = 10;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'x' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:10: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int y = 5;
    y += x;  // x is not the accumulator, y is
  }
}
