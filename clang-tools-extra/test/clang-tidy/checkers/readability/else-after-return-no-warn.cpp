// RUN: %check_clang_tidy %s readability-else-after-return %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-else-after-return.WarnOnUnfixable: false, \
// RUN:     }}'

int h(int);

int lifeTimeExtensionTests(int a) {
  if (a > 0) {
    return a;
  } else {
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
    int b = 0;
    h(b);
  }
  if (int b = a) {
    return a;
  } else {
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
    b++;
  }
  if (int b = a) { // comment-0
    // CHECK-FIXES:      {{^}}  int b = a;
    // CHECK-FIXES-NEXT: {{^}}if (b) { // comment-0
    return a;
  } else { // comment-0
           // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
           // CHECK-FIXES: {{^}}  } // comment-0
    return b;
  }
}
