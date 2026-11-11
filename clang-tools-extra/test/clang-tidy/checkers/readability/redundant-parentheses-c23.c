// RUN: %check_clang_tidy -std=c23-or-later %s readability-redundant-parentheses %t

void typeofOperand(void) {
  typeof(1) a;
  typeof_unqual(1) b;
  typeof(a) c;
  typeof((2)) d;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    typeof(2) d;
  const typeof(a) e = a;
  volatile typeof_unqual(1) f;
  typeof(a) const g = a;
  const typeof((3)) h = 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    const typeof(3) h = 3;
}
