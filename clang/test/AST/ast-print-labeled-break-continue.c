// RUN: %clang_cc1 -std=c2y -ast-print %s | FileCheck %s

void TestLabeledBreakContinue() {
  a: while (true) {
    break a;
    continue a;
    c: for (;;) {
      break a;
      continue a;
      break c;
    }
  }
}

// CHECK-LABEL: void TestLabeledBreakContinue(void) {
// CHECK-NEXT:   a:
// CHECK-NEXT:     while (true)
// CHECK-NEXT:         {
// CHECK-NEXT:             break a;
// CHECK-NEXT:             continue a;
// CHECK-NEXT:           c:
// CHECK-NEXT:             for (;;) {
// CHECK-NEXT:                 break a;
// CHECK-NEXT:                 continue a;
// CHECK-NEXT:                 break c;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT: }
