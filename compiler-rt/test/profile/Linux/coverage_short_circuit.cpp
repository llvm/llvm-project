// REQUIRES: lld-available
// RUN: %clangxx_profgen -std=c++17 -fuse-ld=lld -fcoverage-mapping -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile=%t.profdata 2>&1 | FileCheck %s

void foo() {          // CHECK:       [[@LINE]]| 1|void foo() {
  bool cond1 = false; // CHECK-NEXT:  [[@LINE]]| 1|  bool cond1 = false;
  bool cond2 = true;  // CHECK-NEXT:  [[@LINE]]| 1|  bool cond2 = true;
  if (cond1 &&        // CHECK-NEXT:  [[@LINE]]| 1|  if (cond1 &&
      cond2) {        // CHECK-NEXT:  [[@LINE]]| 0|      cond2) {
  } // CHECK-NEXT:  [[@LINE]]| 0|  }
} // CHECK-NEXT:  [[@LINE]]| 1|}

void bar() {          // CHECK:       [[@LINE]]| 1|void bar() {
  bool cond1 = true;  // CHECK-NEXT:  [[@LINE]]| 1|  bool cond1 = true;
  bool cond2 = false; // CHECK-NEXT:  [[@LINE]]| 1|  bool cond2 = false;
  if (cond1 &&        // CHECK-NEXT:  [[@LINE]]| 1|  if (cond1 &&
      cond2) {        // CHECK-NEXT:  [[@LINE]]| 1|      cond2) {
  } // CHECK-NEXT:  [[@LINE]]| 0|  }
} // CHECK-NEXT:  [[@LINE]]| 1|}

void baz() {          // CHECK:       [[@LINE]]| 1|void baz() {
  bool cond1 = false; // CHECK-NEXT:  [[@LINE]]| 1|  bool cond1 = false;
  bool cond2 = true;  // CHECK-NEXT:  [[@LINE]]| 1|  bool cond2 = true;
  if (cond1           // CHECK-NEXT:  [[@LINE]]| 1|  if (cond1
      &&              // CHECK-NEXT:  [[@LINE]]| 0|      &&
      cond2) {        // CHECK-NEXT:  [[@LINE]]| 0|      cond2) {
  } // CHECK-NEXT:  [[@LINE]]| 0|  }
} // CHECK-NEXT:  [[@LINE]]| 1|}

int main() {
  foo();
  bar();
  baz();
  return 0;
}
