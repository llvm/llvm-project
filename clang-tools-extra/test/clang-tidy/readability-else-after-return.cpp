// RUN: %check_clang_tidy %s readability-else-after-return %t -- -- -std=c++11 -fexceptions

void f(int a) {
  if (a > 0)
    return;
  else // comment-0
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'else' after 'return'
  // CHECK-FIXES: {{^}}  // comment-0
    return;

  if (a > 0) {
    return;
  } else { // comment-1
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
  // CHECK-FIXES: {{^}}  } // comment-1
    return;
  }

  if (a > 0) {
    f(0);
    if (a > 10)
      return;
  } else {
    return;
  }

  if (a > 0)
    f(0);
  else if (a > 10)
    return;
  else // comment-2
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'else' after 'return'
  // CHECK-FIXES: {{^}}  // comment-2
    f(0);
}

void foo() {
  for (unsigned x = 0; x < 42; ++x) {
    if (x) {
      continue;
    } else { // comment-3
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not use 'else' after 'continue'
    // CHECK-FIXES: {{^}}    } // comment-3
      x++;
    }
    if (x) {
      break;
    } else { // comment-4
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not use 'else' after 'break'
    // CHECK-FIXES: {{^}}    } // comment-4
      x++;
    }
    if (x) {
      throw 42;
    } else { // comment-5
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not use 'else' after 'throw'
    // CHECK-FIXES: {{^}}    } // comment-5
      x++;
    }
  }
}
