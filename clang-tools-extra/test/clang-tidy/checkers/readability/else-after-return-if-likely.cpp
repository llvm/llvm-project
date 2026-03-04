// RUN: %check_clang_tidy %s readability-else-after-return %t -- -- -std=c++20

void f() {
  if (true) [[likely]] {
    return;
  } else { // comment-0
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
      // CHECK-FIXES: {{^}}     } // comment-0
  }

  if (false) [[unlikely]] {
    return;
  } else { // comment-1
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
      // CHECK-FIXES: {{^}}     } // comment-1
  }

  if (true) [[likely]]
    return;
  else // comment-2
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'else' after 'return'
    // CHECK-FIXES: {{^}}   // comment-2
    int _ = 10;
}
