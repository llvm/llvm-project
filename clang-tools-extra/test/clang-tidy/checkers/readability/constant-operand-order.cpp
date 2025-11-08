// RUN: %check_clang_tidy %s readability-constant-operand-order %t -- -- -std=c++17
// RUN: %check_clang_tidy %s readability-constant-operand-order %t -- -check-suffixes=LEFT -- -config="{CheckOptions:[{key: readability-constant-operand-order.PreferredConstantSide, value: Left}, {key: readability-constant-operand-order.BinaryOperators, value: '==,!='}]}" -- -std=c++17

void swap_eq(int a) {
  if (0 == a) {} // CHECK-MESSAGES: warning: constant operand should be on the Right side
                 // CHECK-FIXES: if (a == 0) {} //
}

void swap_asym(int a) {
  if (0 < a) {}  // CHECK-MESSAGES: warning:
                 // CHECK-FIXES: if (a > 0) {}  //
}

void null_ptr(int *p) {
  if (nullptr == p) {} // CHECK-MESSAGES: warning:
                       // CHECK-FIXES: if (p == nullptr) {} //
}

// No-fix when side effects:
int g();
void side_effects(int a) {
  if (0 == g()) {} // CHECK-MESSAGES: warning:
                   // CHECK-FIXES-NOT: if (g() == 0)
}

// Config variant: allow Left and only ==,!=
void left_ok(int a) {
  if (0 == a) {} // CHECK-MESSAGES: warning:
                 // CHECK-FIXES: if (a == 0) {} //
                 // CHECK-LEFT-MESSAGES-NOT: readability-constant-operand-order
}
