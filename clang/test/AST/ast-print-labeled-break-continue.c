// RUN: %clang_cc1 -std=c2y -ast-print %s | FileCheck %s

void TestLabeledBreakContinue() {
  a: b: while (true) {
    break a;
    continue b;
    c: for (;;) {
      break a;
      continue b;
      break c;
    }
  }
}

// CHECK-LABEL: void TestLabeledBreakContinue(void) {
// CHECK-NEXT:   a:
// CHECK-NEXT:   b:
// CHECK-NEXT:     while (true)
// CHECK-NEXT:         {
// CHECK-NEXT:             break a;
// CHECK-NEXT:             continue b;
// CHECK-NEXT:           c:
// CHECK-NEXT:             for (;;) {
// CHECK-NEXT:                 break a;
// CHECK-NEXT:                 continue b;
// CHECK-NEXT:                 break c;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT: }
