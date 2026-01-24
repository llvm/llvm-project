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

// Variable can be moved to while loop body. should NOT warn
void test_while_loop() {
  int counter = 0;
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

// Variable can be moved to for-loop body. should NOT warn
void test_for_loop_body() {
  int temp = 0;
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

// should NOT warn.
// moving would make initialization conditional
void test_switch_case(int value) {
  int result = 0;
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

// Variable in nested switch - should NOT warn
// moving would make initialization conditional
void test_nested_switch(int outer, int inner) {
  int nested = 0;
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

// Variable used in switch default only - should NOT warn
// moving would make initialization conditional
void test_switch_default_only(int value) {
  int def = 0;
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


// Helper functions for test cases
int calculateLimit() { return 10; }
int getDefaultError() { return -1; }
void riskyOperation() {}
class Exception { public: int getCode() const { return 1; } };
void handleError(int) {}
int expensive() { return 42; }
bool condition = true;
void use(int) {}
int getValue() { return 5; }
void processResult(int) {}
int transform(int x) { return x * 2; }
void doSomething(int) {}
int calculateValue(int x) { return x * x; }
void process(int) {}

// Unary operator accumulator patterns - currently might warn incorrectly

// Post-increment in loop - should NOT warn (accumulator pattern)
void test_post_increment_loop() {
  int counter = 0;  // Should NOT warn - accumulator with post-increment
  for (int i = 0; i < 10; ++i) {
    counter++;
  }
}

// Pre-increment in loop - should NOT warn (accumulator pattern)
void test_pre_increment_loop() {
  int counter = 0;  // Should NOT warn - accumulator with pre-increment
  for (int i = 0; i < 10; ++i) {
    ++counter;
  }
}

// Post-decrement in loop - should NOT warn (accumulator pattern)
void test_post_decrement_loop() {
  int counter = 100;  // Should NOT warn - accumulator with post-decrement
  while (counter > 0) {
    counter--;
  }
}

// Pre-decrement in loop - should NOT warn (accumulator pattern)
void test_pre_decrement_loop() {
  int counter = 100;  // Should NOT warn - accumulator with pre-decrement
  while (counter > 0) {
    --counter;
  }
}

// Container accumulation patterns - should NOT warn

// Array-like accumulation - should NOT warn
void test_array_accumulation() {
  int results[10];  // Should NOT warn - array accumulator
  int index = 0;    // Should NOT warn - index accumulator
  for (int i = 0; i < 10; ++i) {
    results[index++] = i;
  }
}

// Simple string accumulation - should NOT warn
void test_simple_string_accumulation() {
  char message[100] = "";  // Should NOT warn - string accumulator
  int len = 0;             // Should NOT warn - length accumulator
  for (int i = 0; i < 5; ++i) {
    message[len++] = 'A' + i;
  }
}

// Bitwise accumulation patterns - some already handled, some might not be

// Bitwise OR with compound assignment - should NOT warn (already handled)
void test_bitwise_compound() {
  int flags = 0;  // Should NOT warn - compound assignment accumulator
  for (int i = 0; i < 8; ++i) {
    flags |= (1 << i);
  }
}

// Bitwise OR with explicit assignment - should NOT warn
void test_bitwise_explicit() {
  int flags = 0;  // Should NOT warn - bitwise accumulator pattern
  for (int i = 0; i < 8; ++i) {
    flags = flags | (1 << i);
  }
}

// Bitwise AND accumulation - should NOT warn
void test_bitwise_and() {
  int mask = 0xFF;  // Should NOT warn - bitwise accumulator pattern
  for (int i = 0; i < 8; ++i) {
    mask = mask & ~(1 << i);
  }
}

// Scope reduction opportunities the checker might miss

// Variable used only in loop condition - might be movable
void test_loop_condition_only() {
  int limit = calculateLimit();  // Might be movable to for-loop init
  for (int i = 0; i < limit; ++i) {
    // body doesn't use limit
    doSomething(i);
  }
}

// Variable in exception handling - should warn
void test_exception_handling() {
  int errorCode = getDefaultError();
  // Should warn - errorCode could be moved to catch block
  try {
    riskyOperation();
  } catch (const Exception& e) {
    errorCode = e.getCode();
    handleError(errorCode);
  }
}

// Complex initialization dependencies
void test_initialization_chain() {
  int a = expensive();
  int b = a * 2;  // b could potentially be moved if only used in smaller scope
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'b' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:9: note: used here
  // CHECK-NOTES: :[[@LINE+2]]:18: note: can be declared in this scope
  // Should warn for b - it could be moved to if-block
  if (condition) {
    use(b);
  }
}

// Variable used in nested function calls
void test_nested_calls() {
  int temp = getValue();  // Should NOT warn - initialized with function call
  if (condition) {
    processResult(transform(temp));
  }
}

// Multiple assignment patterns in same loop
void test_multiple_assignments() {
  int sum = 0;     // Should NOT warn - accumulator
  int count = 0;   // Should NOT warn - accumulator
  for (int i = 0; i < 10; ++i) {
    sum += i;
    count++;
  }
}

// Mixed accumulator and non-accumulator usage
void test_mixed_usage_complex() {
  int value = 0;   // Complex case - accumulator in loop, then used elsewhere
  for (int i = 0; i < 5; ++i) {
    value += i;    // Accumulator usage
  }
  if (condition) {
    value = 100;   // Non-accumulator usage
    process(value);
  }
}

// Variable modified in loop but not accumulator pattern
void test_non_accumulator_modification() {
  int temp = 0;
  // Should warn - temp is modified but not in accumulator pattern
  for (int i = 0; i < 10; ++i) {
    temp = i * 2;  // Overwrites previous value, not accumulating
    use(temp);
  }
}

// Accumulator with function calls
void test_accumulator_with_calls() {
  int total = 0;   // Should NOT warn - accumulator pattern with function calls
  for (int i = 0; i < 10; ++i) {
    total += calculateValue(i);
  }
}

// Conditional accumulation
void test_conditional_accumulation() {
  int sum = 0;     // Should NOT warn - conditional accumulator
  for (int i = 0; i < 10; ++i) {
    if (i % 2 == 0) {
      sum += i;
    }
  }
}


// Unary operators outside loops - should NOT warn
void test_unary_outside_loop() {
  int value = 10;
  // Should NOT warn - moving would change semantics (loses initialization)
  if (true) {
    value++;
  }
}

// Pre-increment - should NOT warn
void test_pre_increment_outside_loop() {
  int value = 10;
  // Should NOT warn - moving would lose initialization value
  if (true) {
    ++value;
  }
}

// Post-decrement - should NOT warn  
void test_post_decrement_outside_loop() {
  int counter = 100;
  // Should NOT warn - moving would lose initialization value
  if (true) {
    counter--;
  }
}

// Pre-decrement - should NOT warn
void test_pre_decrement_outside_loop() {
  int counter = 100;
  // Should NOT warn - moving would lose initialization value
  if (true) {
    --counter;
  }
}

// Complex initialization with unary operator - should NOT warn
void test_complex_init_with_unary() {
  int calculated = 5 * 3 + 2;
  // Should NOT warn - moving would lose initialization value
  if (true) {
    calculated++;
  }
}

// Array initialization with unary operator - should NOT warn
void test_array_init_with_unary() {
  int arr[] = {1, 2, 3};
  int size = 3;
  // Should NOT warn - moving would lose initialization value
  if (true) {
    size--;
  }
}

// Pointer initialization with unary operator - should NOT warn
void test_pointer_init_with_unary() {
  int value = 42;
  int* ptr = &value;
  // Should NOT warn - moving would lose initialization value  
  if (true) {
    ++ptr;
  }
}

// Switch body edge cases

// Switch with no default case - should NOT warn
void test_switch_no_default(int value) {
  int result = 0;
  // Should NOT warn - moving to switch body would make initialization conditional
  switch (value) {
    case 1:
      result = 10;
      break;
    case 2:
      result = 20;
      break;
  }
}

// Switch with only default case - should NOT warn
void test_switch_default_only_v2(int value) {
  int result = 0;
  // Should NOT warn - moving to switch body would make initialization conditional
  switch (value) {
    default:
      result = 100;
      break;
  }
}

// Empty switch - should NOT warn (though variable unused)
void test_switch_empty(int value) {
  int result = 0;
  // Should NOT warn - variable not used, but if it were used in switch body, would be conditional
  switch (value) {
  }
}

// Combination cases

// Unary operator inside switch case - should NOT warn
void test_unary_in_switch_case(int value) {
  int counter = 0;
  // Should NOT warn - moving to switch body would make initialization conditional
  switch (value) {
    case 1:
      counter++;
      break;
    default:
      break;
  }
}

// Unary operator inside loop (accumulator pattern) - should NOT warn
void test_unary_in_loop_accumulator() {
  int counter = 0;
  // Should NOT warn - accumulator pattern in loop
  for (int i = 0; i < 10; ++i) {
    counter++;
  }
}

// Multiple unary operations on same variable - should NOT warn
void test_multiple_unary_operations() {
  int value = 10;
  // Should NOT warn - moving would lose initialization value
  if (true) {
    value++;
    ++value;
    value--;
  }
}

// Unary operator with other operations - should NOT warn
void test_unary_with_other_ops() {
  int value = 5;
  // Should NOT warn - moving would lose initialization value
  if (true) {
    value++;
    value *= 2;
  }
}

// Nested switch with unary operator - should NOT warn
void test_nested_switch_with_unary(int outer, int inner) {
  int counter = 0;
  // Should NOT warn - moving to outer switch would make initialization conditional
  switch (outer) {
    case 1:
      switch (inner) {
        case 1:
          counter++;
          break;
      }
      break;
  }
}

// Test cases for member function call initializers - should NOT suggest for-loop initialization
// These test the fix for cases where B.buildUnmerge() was incorrectly flagged for for-loop init

class Builder {
public:
  int buildUnmerge(int type, int reg);
  int getNumOperands();
};

// Member function call in initializer - should NOT warn for for-loop initialization
void test_member_function_call_initializer() {
  Builder B;
  int Reg = 42;
  
  // Should NOT suggest moving to for-loop initialization
  // B.buildUnmerge() is a member function call and too complex for for-loop init
  auto Unmerge = B.buildUnmerge(32, Reg);
  for (int I = 0, E = Unmerge - 1; I != E; ++I) {
    // use I in loop body
    int temp = I * 2;
  }
}

// Similar case with method chaining - should NOT warn
void test_method_chaining() {
  Builder B;
  int Reg = 42;
  
  // Should NOT suggest moving to for-loop initialization  
  // Method call is too complex for for-loop init
  auto Result = B.buildUnmerge(32, Reg);
  for (int I = 0; I < 10; ++I) {
    int value = Result + I;
  }
}

// Regular function call - should NOT warn (existing protection)
int regularFunction(int x);
void test_regular_function_call() {
  int input = 5;
  
  // Should NOT suggest moving to for-loop initialization
  auto result = regularFunction(input);
  for (int I = 0; I < result; ++I) {
    int temp = I;
  }
}

// Simple initialization - should warn for for-loop initialization
void test_simple_initialization_control() {
  int limit = 10;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'limit' can be declared in for-loop initialization
  // CHECK-NOTES: :[[@LINE+1]]:3: note: can be declared in this for-loop
  for (int I = 0; I < limit; ++I) {
    int temp = I;
  }
}

// Test cases for for-loop increment modification - should NOT suggest for-loop initialization
// These test the fix for variables modified in for-loop increment expressions

struct Node {
  Node* getNext();
};

// Variable initialized and modified in for-loop increment - should NOT warn
void test_for_loop_increment_modification() {
  Node root;

  // Should NOT suggest moving to for-loop initialization
  // M is initialized to &root, then modified in for-loop increment
  // Moving would lose the initialization value
  Node* M = &root;
  for (; M; M = M->getNext()) {
    // use M in loop body
    if (M) {
      // process node
    }
  }
}

// Similar case with different initialization - should NOT warn
void test_for_loop_increment_modification_v2() {
  Node nodes[10];

  // Should NOT suggest moving to for-loop initialization
  // ptr is initialized to nodes, then modified in increment
  Node* ptr = nodes;
  for (; ptr; ptr = ptr->getNext()) {
    // process ptr
  }
}

// Variable modified in for-loop increment but dependencies prevent moving - should NOT warn
void test_for_loop_increment_uninitialized() {
  Node root;

  // Should NOT suggest moving root or current to for-loop initialization
  // root: address taken in for-loop init (&root)
  // current: modified in for-loop increment
  Node* current;
  for (current = &root; current; current = current->getNext()) {
    // use current
  }
}

// Variable used in for-loop but not modified in increment - should warn
void test_for_loop_not_modified_in_increment() {
  Node root;

  // Should suggest moving to for-loop initialization
  // node is used in condition but not modified in increment
  Node* node = &root;
  // CHECK-NOTES: :[[@LINE-1]]:9: warning: variable 'node' can be declared in for-loop initialization
  // CHECK-NOTES: :[[@LINE+1]]:3: note: can be declared in this for-loop
  for (int i = 0; node && i < 10; ++i) {
    // use node but don't modify it in increment
    if (node) {
      // process
    }
  }
}

// =============================================================================
// C-SPECIFIC CONSTRUCTS
// =============================================================================

// C-style array with scope reduction opportunity
void test_c_style_array() {
  int arr[10];
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'arr' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+4]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:10: note: can be declared in this scope
  if (1) {
    arr[0] = 42;
    arr[1] = 43;
  }
}

// Function pointer usage
int add(int a, int b) { return a + b; }
void test_function_pointer() {
  int (*func_ptr)(int, int) = add;
  // CHECK-NOTES: :[[@LINE-1]]:9: warning: variable 'func_ptr' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:18: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:10: note: can be declared in this scope
  if (1) {
    int result = func_ptr(1, 2);
  }
}

// C-style struct initialization
struct Point {
  int x, y;
};

void test_c_struct_init() {
  struct Point p = {10, 20};
  // CHECK-NOTES: :[[@LINE-1]]:16: warning: variable 'p' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:15: note: used here
  // CHECK-NOTES: :[[@LINE+3]]:21: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:10: note: can be declared in this scope
  if (1) {
    int sum = p.x + p.y;
  }
}

// goto statement with variable scope
void test_goto_statement() {
  int value = 42;
  // Should NOT warn - used across goto boundary
  if (value > 0) {
    goto end;
  }
  value = 0;
end:
  return;
}

// =============================================================================
// MODERN C++ FEATURES
// =============================================================================

// auto type deduction edge cases
void test_auto_deduction() {
  auto value = 42;
  // CHECK-NOTES: :[[@LINE-1]]:8: warning: variable 'value' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:19: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int doubled = value * 2;
  }
}

// auto with complex type deduction - simplified without std::vector
void test_auto_complex() {
  auto value = 42;
  // CHECK-NOTES: :[[@LINE-1]]:8: warning: variable 'value' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:19: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int doubled = value * 2;
  }
}

// Structured bindings - simplified without std::pair
struct SimplePair {
  int first, second;
};

void test_structured_bindings() {
  SimplePair pair_val = {1, 2};
  // CHECK-NOTES: :[[@LINE-1]]:14: warning: variable 'pair_val' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:15: note: used here
  // CHECK-NOTES: :[[@LINE+3]]:32: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int sum = pair_val.first + pair_val.second;
  }
}

// constexpr variables
void test_constexpr_variable() {
  constexpr int compile_time_val = 42;
  // CHECK-NOTES: :[[@LINE-1]]:17: warning: variable 'compile_time_val' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:18: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int result = compile_time_val * 2;
  }
}

// if constexpr (C++17)
template<bool B>
void test_if_constexpr() {
  int value = 10;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'value' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:19: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:20: note: can be declared in this scope
  if constexpr (B) {
    int doubled = value * 2;
  }
}

// =============================================================================
// COMPLEX CONTROL FLOW
// =============================================================================

// Switch with fallthrough cases
void test_switch_fallthrough(int value) {
  int result = 0;
  // Should NOT warn - used across multiple cases with fallthrough
  switch (value) {
    case 1:
      result = 10;
      // fallthrough
    case 2:
      result += 5;
      break;
    default:
      break;
  }
}

// Nested try-catch - simplified without std::runtime_error
void test_nested_try_catch() {
  int error_code = 0;
  // Should NOT warn - used in multiple exception contexts
  try {
    try {
      error_code = 100;
      throw 42; // throw int instead of std::runtime_error
    } catch (int) {
      error_code = 200;
      throw;
    }
  } catch (...) {
    error_code = 300;
  }
}

// Loop with break/continue affecting scope
void test_loop_break_continue() {
  int counter = 0;
  // Should NOT warn - counter used across break/continue boundaries
  for (int i = 0; i < 10; ++i) {
    if (i % 2 == 0) {
      counter++;
      continue;
    }
    if (counter > 5) {
      break;
    }
    counter += 2;
  }
}

// Nested loops with different variable usage
void test_nested_loop_patterns() {
  int outer_var = 0;
  int inner_var = 0;
  // outer_var: should NOT warn
  // inner_var: should NOT warn
  for (int i = 0; i < 5; ++i) {
    outer_var += i;
    for (int j = 0; j < 3; ++j) {
      inner_var = j * 2;
      int temp = inner_var + 1;
    }
  }
}

// =============================================================================
// VARIABLE LIFETIME EDGE CASES
// =============================================================================

// RAII pattern with destructor
class Resource {
public:
  Resource() {}
  ~Resource() {}
  void use() {}
};

void test_raii_pattern() {
  Resource res;
  // CHECK-NOTES: :[[@LINE-1]]:12: warning: variable 'res' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:5: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    res.use();
  }
  // Destructor called here
}

// Static local variable in different contexts
void test_static_local_contexts() {
  static int call_count = 0;
  // Should NOT warn - static variables have different lifetime semantics
  if (true) {
    call_count++;
  }
}

// Thread-local variable (C++11)
thread_local int tls_var = 0;
void test_thread_local() {
  int local_copy = tls_var;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'local_copy' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:19: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int doubled = local_copy * 2;
  }
}

// =============================================================================
// PREPROCESSOR INTERACTIONS
// =============================================================================

#define USE_VAR(x) ((x) * 2)

// Variable used through macro
void test_macro_usage() {
  int value = 10;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'value' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:26: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int result = USE_VAR(value);
  }
}

// Conditional compilation
// Should NOT warn - variable only used in conditional block
void test_conditional_compilation_undefined() {
  int debug_var = 42;
#ifdef DEBUG
  if (true) {
    int temp = debug_var;
  }
#endif
}

#define DEBUG_DEFINED
// Should warn - variable used in conditional block
void test_conditional_compilation_defined() {
  int debug_var = 42;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'debug_var' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+4]]:16: note: used here
  // CHECK-NOTES: :[[@LINE+2]]:13: note: can be declared in this scope
#ifdef DEBUG_DEFINED
  if (true) {
    int temp = debug_var;
  }
#endif
}

// =============================================================================
// ADDITIONAL EDGE CASES
// =============================================================================

// Variable declared in one scope, used in sibling scope
void test_sibling_scopes() {
  int shared = 0;
  // Should NOT warn - used across sibling scopes
  if (true) {
    shared = 10;
  } else {
    shared = 20;
  }
}

// Variable with comma operator
void test_comma_operator() {
  int a = 1, b = 2;
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'a' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+6]]:15: note: used here
  // CHECK-NOTES: :[[@LINE+4]]:13: note: can be declared in this scope
  // CHECK-NOTES: :[[@LINE-4]]:14: warning: variable 'b' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:19: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int sum = a + b;
  }
}

// Variable in ternary operator
void test_ternary_operator() {
  // Should warn for true_val and false_val - only used in ternary
  int condition_var = 1;
  int true_val = 10;
  int false_val = 20;
  // CHECK-NOTES: :[[@LINE-3]]:7: warning: variable 'condition_var' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+9]]:18: note: used here
  // CHECK-NOTES: :[[@LINE+7]]:13: note: can be declared in this scope
  // CHECK-NOTES: :[[@LINE-5]]:7: warning: variable 'true_val' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+6]]:34: note: used here
  // CHECK-NOTES: :[[@LINE+4]]:13: note: can be declared in this scope
  // CHECK-NOTES: :[[@LINE-7]]:7: warning: variable 'false_val' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:45: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int result = condition_var ? true_val : false_val;
  }
}

void test_ternary_operator_cond(int cond) {
  // Should warn for true_val and false_val - only used in ternary
  int true_val = 10;
  int false_val = 20;
  // CHECK-NOTES: :[[@LINE-2]]:7: warning: variable 'true_val' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+6]]:25: note: used here
  // CHECK-NOTES: :[[@LINE+4]]:13: note: can be declared in this scope
  // CHECK-NOTES: :[[@LINE-4]]:7: warning: variable 'false_val' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:36: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int result = cond ? true_val : false_val;
  }
}

// Variable used in sizeof expression
void test_sizeof_usage() {
  // Should warn
  int array[100];
  // CHECK-NOTES: :[[@LINE-1]]:7: warning: variable 'array' can be declared in a smaller scope
  // CHECK-NOTES: :[[@LINE+3]]:23: note: used here
  // CHECK-NOTES: :[[@LINE+1]]:13: note: can be declared in this scope
  if (true) {
    int size = sizeof(array);
  }
}

// Variable used in decltype (C++11)
void test_decltype_usage() {
  int value = 42;
  // Should NOT warn - decltype doesn't evaluate the expression
  if (true) {
    decltype(value) another = 10;
  }
}
