// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -std=c++20 %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

struct A { ~A(); };
void range_for_init() {
  for (A a; int x : (int[]){1, 2, 3}) {}
}

// CHECK-LABEL: void range_for_init()
// CHECK:       [B6 (ENTRY)]
// CHECK-NEXT:    Succs (1): B5

// CHECK:       [B1]
// CHECK-NEXT:    1: [B5.2].~A() (Implicit destructor)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:    1: __begin1
// CHECK-NEXT:    2: [B2.1] (ImplicitCastExpr, LValueToRValue, int *)
// CHECK-NEXT:    3: __end1
// CHECK-NEXT:    4: [B2.3] (ImplicitCastExpr, LValueToRValue, int *)
// CHECK-NEXT:    5: [B2.2] != [B2.4]
// CHECK-NEXT:    T: for (A a; int x : [B5.8]) {
// CHECK-NEXT:  }
// CHECK:         Preds (2): B3 B5
// CHECK-NEXT:    Succs (2): B4 B1

// CHECK:       [B5]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B5.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    3: 1
// CHECK-NEXT:    4: 2
// CHECK-NEXT:    5: 3
// CHECK-NEXT:    6: {[B5.3], [B5.4], [B5.5]}
// CHECK-NEXT:    7: (int[3])[B5.6]
// CHECK-NEXT:    8: [B5.7]
// CHECK-NEXT:    9: auto &&__range1 = (int[3]){1, 2, 3};

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
