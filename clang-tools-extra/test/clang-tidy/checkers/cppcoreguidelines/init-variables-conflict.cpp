// RUN: %check_clang_tidy %s cppcoreguidelines-init-variables,readability-isolate-declaration %t

void foo() {
  int A, B, C;
  // CHECK-MESSAGES-DAG: :[[@LINE-1]]:7: warning: variable 'A' is not initialized
  // CHECK-MESSAGES-DAG: :[[@LINE-2]]:10: warning: variable 'B' is not initialized
  // CHECK-MESSAGES-DAG: :[[@LINE-3]]:13: warning: variable 'C' is not initialized
  // CHECK-MESSAGES-DAG: :[[@LINE-4]]:3: warning: multiple declarations in a single statement reduces readability

  // Only the isolate declarations fix-it should be applied

  //      CHECK-FIXES: int A;
  // CHECK-FIXES-NEXT: int B;
  // CHECK-FIXES-NEXT: int C;
}
