// RUN: %check_clang_tidy -std=c++20-or-later %s readability-else-after-return %t

void g();

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
    // CHECK-FIXES: {{^}}  else // comment-2
    int _ = 10;

  if (false)
    return;
  else [[likely]] { // comment-3
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'else' after 'return'
    // CHECK-FIXES: {{^}}  {{[[][[]}}likely{{[]][]]}} { // comment-3
    int _ = 31;
  }

  if (true)
    return;
  else [[unlikely]]  // comment-4
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'else' after 'return'
    // CHECK-FIXES: {{^}}  {{[[][[]}}unlikely{{[]][]]}} // comment-4
    g();

  if (false)
    [[clang::musttail]] return f();
  else // comment-5
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'else' after 'return'
    // CHECK-FIXES: {{^}}  // comment-5
    g();

  if (false) [[likely]]
    [[clang::musttail]] return f();
  else // comment-6
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'else' after 'return'
    // CHECK-FIXES: {{^}}  // comment-6
    g();

  if (false) [[likely]] {
    [[clang::musttail]] return f();
  } else { // comment-7
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
    // CHECK-FIXES: {{^}}  } // comment-7
    g();
  }
}
