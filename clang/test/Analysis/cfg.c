// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -Wno-error=invalid-gnu-asm-cast %s > %t 2>&1
// RUN: FileCheck --input-file=%t --check-prefix=CHECK %s

// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c2y -Wno-error=invalid-gnu-asm-cast %s > %t 2>&1
// RUN: FileCheck --input-file=%t --check-prefixes=CHECK,SINCE-C26 %s

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

#if __STDC_VERSION__ >= 202400L // If C26 or above
// SINCE-C26:      int labeled_break_and_continue(int x)
// SINCE-C26-NEXT:  [B17 (ENTRY)]
// SINCE-C26-NEXT:    Succs (1): B2
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B1]
// SINCE-C26-NEXT:    1: 0
// SINCE-C26-NEXT:    2: return [B1.1];
// SINCE-C26-NEXT:    Preds (1): B9
// SINCE-C26-NEXT:    Succs (1): B0
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B2]
// SINCE-C26-NEXT:   a:
// SINCE-C26-NEXT:    1: x
// SINCE-C26-NEXT:    2: [B2.1] (ImplicitCastExpr, LValueToRValue, int)
// SINCE-C26-NEXT:    T: switch [B2.2]
// SINCE-C26-NEXT:    Preds (1): B17
// SINCE-C26-NEXT:    Succs (3): B9 B16 B8
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B3]
// SINCE-C26-NEXT:    1: x
// SINCE-C26-NEXT:    2: [B3.1] (ImplicitCastExpr, LValueToRValue, int)
// SINCE-C26-NEXT:    3: 2
// SINCE-C26-NEXT:    4: [B3.2] + [B3.3]
// SINCE-C26-NEXT:    5: return [B3.4];
// SINCE-C26-NEXT:    Preds (3): B6 B7 B4
// SINCE-C26-NEXT:    Succs (1): B0
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B4]
// SINCE-C26-NEXT:   c:
// SINCE-C26-NEXT:    1: x
// SINCE-C26-NEXT:    2: [B4.1] (ImplicitCastExpr, LValueToRValue, int)
// SINCE-C26-NEXT:    T: switch [B4.2]
// SINCE-C26-NEXT:    Preds (1): B8
// SINCE-C26-NEXT:    Succs (3): B6 B7 B3
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B5]
// SINCE-C26-NEXT:    1: x
// SINCE-C26-NEXT:    2: [B5.1] (ImplicitCastExpr, LValueToRValue, int)
// SINCE-C26-NEXT:    3: 3
// SINCE-C26-NEXT:    4: [B5.2] + [B5.3]
// SINCE-C26-NEXT:    5: return [B5.4];
// SINCE-C26-NEXT:    Succs (1): B0
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B6]
// SINCE-C26-NEXT:   case 30:
// SINCE-C26-NEXT:    T: break c;
// SINCE-C26-NEXT:    Preds (1): B4
// SINCE-C26-NEXT:    Succs (1): B3
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B7]
// SINCE-C26-NEXT:   case 10:
// SINCE-C26-NEXT:    T: break a;
// SINCE-C26-NEXT:    Preds (1): B4
// SINCE-C26-NEXT:    Succs (1): B3
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B8]
// SINCE-C26-NEXT:   default:
// SINCE-C26-NEXT:    Preds (1): B2
// SINCE-C26-NEXT:    Succs (1): B4
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B9]
// SINCE-C26-NEXT:   case 2:
// SINCE-C26-NEXT:    T: break a;
// SINCE-C26-NEXT:    Preds (2): B2 B11
// SINCE-C26-NEXT:    Succs (1): B1
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B10]
// SINCE-C26-NEXT:    1: 1
// SINCE-C26-NEXT:    T: do ... while [B10.1]
// SINCE-C26-NEXT:    Preds (1): B12
// SINCE-C26-NEXT:    Succs (2): B14 NULL
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B11]
// SINCE-C26-NEXT:    T: break b;
// SINCE-C26-NEXT:    Preds (1): B13
// SINCE-C26-NEXT:    Succs (1): B9
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B12]
// SINCE-C26-NEXT:    1: x
// SINCE-C26-NEXT:    2: ++[B12.1]
// SINCE-C26-NEXT:    T: continue b;
// SINCE-C26-NEXT:    Preds (1): B13
// SINCE-C26-NEXT:    Succs (1): B10
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B13]
// SINCE-C26-NEXT:    1: x
// SINCE-C26-NEXT:    2: [B13.1] (ImplicitCastExpr, LValueToRValue, int)
// SINCE-C26-NEXT:    3: x
// SINCE-C26-NEXT:    4: [B13.3] (ImplicitCastExpr, LValueToRValue, int)
// SINCE-C26-NEXT:    5: [B13.2] * [B13.4]
// SINCE-C26-NEXT:    6: 100
// SINCE-C26-NEXT:    7: [B13.5] > [B13.6]
// SINCE-C26-NEXT:    T: if [B13.7]
// SINCE-C26-NEXT:    Preds (2): B14 B15
// SINCE-C26-NEXT:    Succs (2): B12 B11
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B14]
// SINCE-C26-NEXT:    Preds (1): B10
// SINCE-C26-NEXT:    Succs (1): B13
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B15]
// SINCE-C26-NEXT:   b:
// SINCE-C26-NEXT:    Preds (1): B16
// SINCE-C26-NEXT:    Succs (1): B13
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B16]
// SINCE-C26-NEXT:   case 1:
// SINCE-C26-NEXT:    Preds (1): B2
// SINCE-C26-NEXT:    Succs (1): B15
// SINCE-C26-EMPTY:
// SINCE-C26-NEXT:  [B0 (EXIT)]
// SINCE-C26-NEXT:    Preds (3): B1 B3 B5
int labeled_break_and_continue(int x) {
  a: switch (x) {
    case 1:
      b: do {
        if (x * x > 100) {
          ++x;
          continue b;
        }
        break b;
      } while (1);
    case 2:
      break a;
    default:
    c: switch (x) {
      case 10:
        break a;
      case 30:
        break c;
      return x + 3; // dead code
    }
    return x + 2;
  }

  return 0;
}

#endif // __STDC_VERSION__ >= 202400L // If C26 or above
