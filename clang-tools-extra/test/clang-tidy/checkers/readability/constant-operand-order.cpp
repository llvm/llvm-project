void ok_eq(int a) {
  if (a == 0) {} // CHECK-MESSAGES-NOT: readability-constant-operand-order
                 // CHECK-LEFT-MESSAGES: warning: constant operand should be on the Left side
                 // CHECK-LEFT-FIXES: if (0 == a) {} //
}


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