// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG %s 2>&1 | FileCheck %s

void *target;
int indirectBlockSuccessorDeterminism() {
    (void)&&L1;
    (void)&&L2;
    (void)&&L3;
    (void)&&L4;
    (void)&&L5;
    (void)&&L6;
    (void)&&L7;
    (void)&&L8;
    (void)&&L9;
    (void)&&L10;
    (void)&&L11;
    (void)&&L12;
    (void)&&L13;
    (void)&&L14;
    (void)&&L15;
    (void)&&L16;
    (void)&&L17;
    (void)&&L18;
    (void)&&L19;
    (void)&&L20;
    (void)&&L21;
    (void)&&L22;
    (void)&&L23;
    (void)&&L24;
    (void)&&L25;
    (void)&&L26;
    (void)&&L27;
    (void)&&L28;
    (void)&&L29;
    (void)&&L30;
    (void)&&L31;
    (void)&&L32;
    (void)&&L33;
    (void)&&L34;
    (void)&&L35;
    (void)&&L36;
    (void)&&L37;
    (void)&&L38;
    (void)&&L39;
    (void)&&L40;

    goto *target;
  L1:
  L2:
  L3:
  L4:
  L5:
  L6:
  L7:
  L8:
  L9:
  L10:
  L11:
  L12:
  L13:
  L14:
  L15:
  L16:
  L17:
  L18:
  L19:
  L20:
  L21:
  L22:
  L23:
  L24:
  L25:
  L26:
  L27:
  L28:
  L29:
  L30:
  L31:
  L32:
  L33:
  L34:
  L35:
  L36:
  L37:
  L38:
  L39:
  L40:
    return 0;
}

// CHECK-LABEL:  [B41 (INDIRECT GOTO DISPATCH)]
// CHECK-NEXT:   Preds (1): B42
// CHECK-NEXT:  Succs (40): B1 B2 B3 B4 B5 B6 B7 B8
// CHECK-NEXT:       B9 B10 B11 B12 B13 B14 B15 B16 B17 B18
// CHECK-NEXT:       B19 B20 B21 B22 B23 B24 B25 B26 B27 B28
// CHECK-NEXT:       B29 B30 B31 B32 B33 B34 B35 B36 B37 B38
// CHECK-NEXT:       B39 B40
