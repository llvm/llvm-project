// RUN: %check_clang_tidy -std=c++17-or-later %s readability-else-after-return %t

// Constexpr if is an exception to the rule, we cannot remove the else.
void f() {
  if (sizeof(int) > 4)
    return;
  else
    return;
  // CHECK-MESSAGES: [[@LINE-2]]:3: warning: do not use 'else' after 'return'

  if constexpr (sizeof(int) > 4)
    return;
  else
    return;

  if constexpr (sizeof(int) > 4)
    return;
  else if constexpr (sizeof(long) > 4)
    return;
  else
    return;
}
