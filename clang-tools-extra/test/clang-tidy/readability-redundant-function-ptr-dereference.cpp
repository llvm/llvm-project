// RUN: %check_clang_tidy %s readability-redundant-function-ptr-dereference %t

void f(int i);

void positive() {
  void (*p)(int) = f;

  (**p)(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: redundant repeated dereference of function pointer [readability-redundant-function-ptr-dereference]
  // CHECK-FIXES: (*p)(1);
  (*****p)(2);
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: redundant repeated
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: redundant repeated
  // CHECK-MESSAGES: :[[@LINE-3]]:6: warning: redundant repeated
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: redundant repeated
  // CHECK-FIXES: (*p)(2);
}

void negative() {
  void (*q)(int) = &f;

  q(1);
  (*q)(2);
}
