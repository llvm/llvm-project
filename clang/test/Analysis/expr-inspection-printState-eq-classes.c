// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=debug.ExprInspection %s 2>&1 | FileCheck %s

void clang_analyzer_printState(void);

void test_equivalence_classes(int a, int b, int c, int d) {
  if (a + b != c)
    return;
  if (a != d)
    return;
  if (b != 0)
    return;
  clang_analyzer_printState();
  (void)(a * b * c * d);
  return;
}

// CHECK:      "equivalence_classes": [
// CHECK-NEXT:     [ "(reg_$0<int a>) != (reg_$2<int c>)" ],
// CHECK-NEXT:     [ "reg_$0<int a>", "reg_$2<int c>", "reg_$3<int d>" ]
// CHECK-NEXT: ],
