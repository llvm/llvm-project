// RUN: %check_clang_tidy -std=c++11-or-later %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing

bool cond(int X = 0);
void sink();

void two_if_chain() {
  // CHECK-NOTES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    // CHECK-NOTES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(1))
      sink();
  }
}

void long_if_chain() {
  // CHECK-NOTES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    // CHECK-NOTES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(1)) {
      // CHECK-NOTES: :[[@LINE+1]]:7: note: nested 'if' statement to merge declared here
      if (cond(2))
        sink();
    }
  }
}
