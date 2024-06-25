// RUN: %check_clang_tidy -std=c++23 %s readability-else-after-return %t 

// Consteval if is an exception to the rule, we cannot remove the else.
void f() {
  if (sizeof(int) > 4) {
    return;
  } else {
    return;
  }
  // CHECK-MESSAGES: [[@LINE-3]]:5: warning: do not use 'else' after 'return'

  if consteval {
    return;
  } else {
    return;
  }
}
