// RUN: %check_clang_tidy %s misc-no-recursion %t

// We don't have the definition of this function,
// so we can't tell anything about it..
void external();

// This function is obviously not recursive.
void no_recursion() {
}

// Since we don't know what `external()` does,
// we don't know if this is recursive or not.
void maybe_no_recursion() {
  external();
}

// Function calls itself - obviously a recursion.
void endless_recursion() {
  endless_recursion();
}

// CHECK-NOTES: :[[@LINE-4]]:6: warning: function 'endless_recursion' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-5]]:6: note: example recursive call chain, starting from function 'endless_recursion'
// CHECK-NOTES: :[[@LINE-5]]:3: note: Frame #1: function 'endless_recursion' calls function 'endless_recursion' here:
// CHECK-NOTES: :[[@LINE-6]]:3: note: ... which was the starting point of the recursive call chain; there may be other cycles

bool external_oracle();
bool another_external_oracle();

// Function calls itself if some external function said so - recursion.
void maybe_endless_recursion() {
  if (external_oracle())
    maybe_endless_recursion();
}

// CHECK-NOTES: :[[@LINE-5]]:6: warning: function 'maybe_endless_recursion' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-6]]:6: note: example recursive call chain, starting from function 'maybe_endless_recursion'
// CHECK-NOTES: :[[@LINE-5]]:5: note: Frame #1: function 'maybe_endless_recursion' calls function 'maybe_endless_recursion' here:
// CHECK-NOTES: :[[@LINE-6]]:5: note: ... which was the starting point of the recursive call chain; there may be other cycles

// Obviously-constrained recursion.
void recursive_countdown(unsigned x) {
  if (x == 0)
    return;
  recursive_countdown(x - 1);
}

// CHECK-NOTES: :[[@LINE-6]]:6: warning: function 'recursive_countdown' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-7]]:6: note: example recursive call chain, starting from function 'recursive_countdown'
// CHECK-NOTES: :[[@LINE-5]]:3: note: Frame #1: function 'recursive_countdown' calls function 'recursive_countdown' here:
// CHECK-NOTES: :[[@LINE-6]]:3: note: ... which was the starting point of the recursive call chain; there may be other cycles

void indirect_recursion();
void conditionally_executed() {
  if (external_oracle())
    indirect_recursion();
}
void indirect_recursion() {
  if (external_oracle())
    conditionally_executed();
}

// CHECK-NOTES: :[[@LINE-9]]:6: warning: function 'conditionally_executed' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-6]]:6: note: example recursive call chain, starting from function 'indirect_recursion'
// CHECK-NOTES: :[[@LINE-5]]:5: note: Frame #1: function 'indirect_recursion' calls function 'conditionally_executed' here:
// CHECK-NOTES: :[[@LINE-10]]:5: note: Frame #2: function 'conditionally_executed' calls function 'indirect_recursion' here:
// CHECK-NOTES: :[[@LINE-11]]:5: note: ... which was the starting point of the recursive call chain; there may be other cycles
// CHECK-NOTES: :[[@LINE-10]]:6: warning: function 'indirect_recursion' is within a recursive call chain [misc-no-recursion]

void taint();
void maybe_selfrecursion_with_two_backedges() {
  if (external_oracle())
    maybe_selfrecursion_with_two_backedges();
  taint();
  if (another_external_oracle())
    maybe_selfrecursion_with_two_backedges();
}

// CHECK-NOTES: :[[@LINE-8]]:6: warning: function 'maybe_selfrecursion_with_two_backedges' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-9]]:6: note: example recursive call chain, starting from function 'maybe_selfrecursion_with_two_backedges'
// CHECK-NOTES: :[[@LINE-8]]:5: note: Frame #1: function 'maybe_selfrecursion_with_two_backedges' calls function 'maybe_selfrecursion_with_two_backedges' here:
// CHECK-NOTES: :[[@LINE-9]]:5: note: ... which was the starting point of the recursive call chain; there may be other cycles

void indirect_recursion_with_alternatives();
void conditionally_executed_choice_0() {
  if (external_oracle())
    indirect_recursion_with_alternatives();
}
void conditionally_executed_choice_1() {
  if (external_oracle())
    indirect_recursion_with_alternatives();
}
void indirect_recursion_with_alternatives() {
  if (external_oracle())
    conditionally_executed_choice_0();
  else
    conditionally_executed_choice_1();
}

// CHECK-NOTES: :[[@LINE-15]]:6: warning: function 'conditionally_executed_choice_0' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-8]]:6: note: example recursive call chain, starting from function 'indirect_recursion_with_alternatives'
// CHECK-NOTES: :[[@LINE-7]]:5: note: Frame #1: function 'indirect_recursion_with_alternatives' calls function 'conditionally_executed_choice_0' here:
// CHECK-NOTES: :[[@LINE-16]]:5: note: Frame #2: function 'conditionally_executed_choice_0' calls function 'indirect_recursion_with_alternatives' here:
// CHECK-NOTES: :[[@LINE-17]]:5: note: ... which was the starting point of the recursive call chain; there may be other cycles
// CHECK-NOTES: :[[@LINE-16]]:6: warning: function 'conditionally_executed_choice_1' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-13]]:6: warning: function 'indirect_recursion_with_alternatives' is within a recursive call chain [misc-no-recursion]

static void indirect_recursion_with_depth2();
static void conditionally_executed_depth1() {
  if (external_oracle())
    indirect_recursion_with_depth2();
}
static void conditionally_executed_depth0() {
  if (external_oracle())
    conditionally_executed_depth1();
}
void indirect_recursion_with_depth2() {
  if (external_oracle())
    conditionally_executed_depth0();
}

// CHECK-NOTES: :[[@LINE-13]]:13: warning: function 'conditionally_executed_depth1' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-10]]:13: note: example recursive call chain, starting from function 'conditionally_executed_depth0'
// CHECK-NOTES: :[[@LINE-9]]:5: note: Frame #1: function 'conditionally_executed_depth0' calls function 'conditionally_executed_depth1' here:
// CHECK-NOTES: :[[@LINE-14]]:5: note: Frame #2: function 'conditionally_executed_depth1' calls function 'indirect_recursion_with_depth2' here:
// CHECK-NOTES: :[[@LINE-7]]:5: note: Frame #3: function 'indirect_recursion_with_depth2' calls function 'conditionally_executed_depth0' here:
// CHECK-NOTES: :[[@LINE-8]]:5: note: ... which was the starting point of the recursive call chain; there may be other cycles
// CHECK-NOTES: :[[@LINE-15]]:13: warning: function 'conditionally_executed_depth0' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-12]]:6: warning: function 'indirect_recursion_with_depth2' is within a recursive call chain [misc-no-recursion]

int boo();
void foo(int x = boo()) {}
void bar() {
  foo();
  foo();
}
int boo() {
  bar();
  return 0;
}

// CHECK-NOTES: :[[@LINE-9]]:6: warning: function 'bar' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-6]]:5: note: example recursive call chain, starting from function 'boo'
// CHECK-NOTES: :[[@LINE-6]]:3: note: Frame #1: function 'boo' calls function 'bar' here:
// CHECK-NOTES: :[[@LINE-13]]:18: note: Frame #2: function 'bar' calls function 'boo' here:
// CHECK-NOTES: :[[@LINE-14]]:18: note: ... which was the starting point of the recursive call chain; there may be other cycles
// CHECK-NOTES: :[[@LINE-10]]:5: warning: function 'boo' is within a recursive call chain [misc-no-recursion]

int recursion_through_function_ptr() {
  auto *ptr = &recursion_through_function_ptr;
  if (external_oracle())
    return ptr();
  return 0;
}

int recursion_through_lambda() {
  auto zz = []() {
    if (external_oracle())
      return recursion_through_lambda();
    return 0;
  };
  return zz();
}

// CHECK-NOTES: :[[@LINE-9]]:5: warning: function 'recursion_through_lambda' is within a recursive call chain [misc-no-recursion]
// CHECK-NOTES: :[[@LINE-9]]:13: note: example recursive call chain, starting from function 'operator()'
// CHECK-NOTES: :[[@LINE-8]]:14: note: Frame #1: function 'operator()' calls function 'recursion_through_lambda' here:
// CHECK-NOTES: :[[@LINE-6]]:10: note: Frame #2: function 'recursion_through_lambda' calls function 'operator()' here:
// CHECK-NOTES: :[[@LINE-7]]:10: note: ... which was the starting point of the recursive call chain; there may be other cycles
// CHECK-NOTES: :[[@LINE-13]]:13: warning: function 'operator()' is within a recursive call chain [misc-no-recursion]

struct recursion_through_destructor {
  ~recursion_through_destructor() {
    if (external_oracle()) {
      recursion_through_destructor variable;
      // variable goes out of scope, it's destructor runs, and we are back here.
    }
  }
};
