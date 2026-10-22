// RUN: clang-refactor -action switch-to-if %s -- 2>&1 | FileCheck %s

void f(int k) {
  switch (k) {
  case 5:
    ping();
    break;
  case 7:
    pong();
    break;
  }
}

// CHECK: if (k == 5) {
// CHECK-NEXT:     ping();
// CHECK-NEXT: } else if (k == 7) {
// CHECK-NEXT:     pong();
// CHECK-NEXT: }
