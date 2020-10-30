// RUN: %clang_profgen -o %t -O3 %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s

void __llvm_profile_reset_counters(void);
void foo(int);
int main(void) {
  foo(0);
  __llvm_profile_reset_counters();
  foo(1);
  return 0;
}
void foo(int N) {
  // CHECK-LABEL: define{{( dso_local)?}} void @foo(
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[FOO:[0-9]+]]
  if (N) {}
}
// CHECK: ![[FOO]] = !{!"branch_weights", i64 2, i64 1}
