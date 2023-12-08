// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -fheinous-gnu-extensions %s > %t 2>&1
// RUN: FileCheck --input-file=%t --check-prefix=CHECK %s

// This file is the C version of cfg.cpp.
// Tests that are C-specific should go into this file.

// CHECK-LABEL: void checkWrap(int i)
// CHECK: ENTRY
// CHECK-NEXT: Succs (1): B1
// CHECK: [B1]
// CHECK: Succs (21): B2 B3 B4 B5 B6 B7 B8 B9
// CHECK: B10 B11 B12 B13 B14 B15 B16 B17 B18 B19
// CHECK: B20 B21 B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT: Preds (21): B2 B3 B4 B5 B6 B7 B8 B9
// CHECK-NEXT: B10 B11 B12 B13 B14 B15 B16 B17 B18 B19
// CHECK-NEXT: B20 B21 B1
void checkWrap(int i) {
  switch(i) {
    case 0: break;
    case 1: break;
    case 2: break;
    case 3: break;
    case 4: break;
    case 5: break;
    case 6: break;
    case 7: break;
    case 8: break;
    case 9: break;
    case 10: break;
    case 11: break;
    case 12: break;
    case 13: break;
    case 14: break;
    case 15: break;
    case 16: break;
    case 17: break;
    case 18: break;
    case 19: break;
  }
}

// CHECK-LABEL: void checkGCCAsmRValueOutput(void)
// CHECK: [B2 (ENTRY)]
// CHECK-NEXT: Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1: int arg
// CHECK-NEXT:   2: arg
// CHECK-NEXT:   3: (int)[B1.2] (CStyleCastExpr, NoOp, int)
// CHECK-NEXT:   4: asm ("" : "=r" ([B1.3]));
// CHECK-NEXT:   5: arg
// CHECK-NEXT:   6: asm ("" : "=r" ([B1.5]));
void checkGCCAsmRValueOutput(void) {
  int arg;
  __asm__("" : "=r"((int)arg));  // rvalue output operand
  __asm__("" : "=r"(arg));       // lvalue output operand
}

// CHECK-LABEL: int overlap_compare(int x)
// CHECK: [B2]
// CHECK-NEXT:   1: 1
// CHECK-NEXT:   2: return [B2.1];
// CHECK-NEXT:   Preds (1): B3(Unreachable)
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B3]
// CHECK-NEXT:   1: x
// CHECK-NEXT:   2: [B3.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 5
// CHECK-NEXT:   4: [B3.2] > [B3.3]
// CHECK-NEXT:   T: if [B4.5] && [B3.4]
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (2): B2(Unreachable) B1
int overlap_compare(int x) {
  if (x == -1 && x > 5)
    return 1;

  return 2;
}

// CHECK-LABEL: void vla_simple(int x)
// CHECK: [B1]
// CHECK-NEXT:   1: x
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: int vla[x];
void vla_simple(int x) {
  int vla[x];
}

// CHECK-LABEL: void vla_typedef(int x)
// CHECK: [B1]
// CHECK-NEXT:   1: x
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: typedef int VLA[x];
void vla_typedef(int x) {
  typedef int VLA[x];
}

// CHECK-LABEL: void vla_typedef_multi(int x, int y)
// CHECK:  [B1]
// CHECK-NEXT:   1: y
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: x
// CHECK-NEXT:   4: [B1.3] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   5: typedef int VLA[x][y];
void vla_typedef_multi(int x, int y) {
  typedef int VLA[x][y];
}

// CHECK-LABEL: void vla_type_indirect(int x)
// CHECK:  [B1]
// CHECK-NEXT:   1: int (*p_vla)[x];
// CHECK-NEXT:   2: void (*fp_vla)(int *);
void vla_type_indirect(int x) {
  // Should evaluate x
  // FIXME: does not work
  int (*p_vla)[x];

  // Do not evaluate x
  void (*fp_vla)(int[x]);
}
