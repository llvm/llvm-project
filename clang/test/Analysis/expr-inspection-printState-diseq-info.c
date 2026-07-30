// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=debug.ExprInspection %s 2>&1 | FileCheck %s

void clang_analyzer_printState(void);

void test_disequality_info(int e0, int b0, int b1, int c0) {
  int e1 = e0 - b0;
  if (b0 == 2) {
    int e2 = e1 - b1;
    if (e2 > 0) {
      if (b1 != c0)
        clang_analyzer_printState();
    }
  }
}

 // CHECK:        "disequality_info": [
 // CHECK-NEXT:     {
 // CHECK-NEXT:       "class": [ "(reg_$0<int e0>) - 2" ],
 // CHECK-NEXT:       "disequal_to": [
 // CHECK-NEXT:         [ "reg_$7<int b1>" ]]
 // CHECK-NEXT:     },
 // CHECK-NEXT:     {
 // CHECK-NEXT:       "class": [ "reg_$15<int c0>" ],
 // CHECK-NEXT:       "disequal_to": [
 // CHECK-NEXT:         [ "reg_$7<int b1>" ]]
 // CHECK-NEXT:     },
 // CHECK-NEXT:     {
 // CHECK-NEXT:       "class": [ "reg_$7<int b1>" ],
 // CHECK-NEXT:       "disequal_to": [
 // CHECK-NEXT:         [ "(reg_$0<int e0>) - 2" ],
 // CHECK-NEXT:         [ "reg_$15<int c0>" ]]
 // CHECK-NEXT:     }
 // CHECK-NEXT:   ],
