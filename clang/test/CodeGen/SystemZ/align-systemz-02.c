// RUN: %clang_cc1 -triple s390x-linux-gnu %s -o - -target-feature +vector -emit-llvm \
// RUN:    | FileCheck %s -check-prefix=VECIR
// RUN: %clang_cc1 -triple s390x-linux-gnu %s -o - -target-feature +vector -emit-obj -S \
// RUN:    | FileCheck %s -check-prefix=VECASM
// RUN: %clang_cc1 -triple s390x-linux-gnu %s -o - -target-feature -vector -emit-llvm \
// RUN:    | FileCheck %s -check-prefix=SCALIR
// RUN: %clang_cc1 -triple s390x-linux-gnu %s -o - -target-feature -vector -emit-obj -S \
// RUN:    | FileCheck %s -check-prefix=SCALASM
// REQUIRES: systemz-registered-target

typedef __attribute__((vector_size(16))) signed int vec_sint;

volatile vec_sint GlobVsi;

struct S {
  int A;
  vec_sint Vsi;
} GlobS;

void fun() {
  GlobS.Vsi = GlobVsi;
}

// VECIR: %struct.S = type { i32, <4 x i32> }
// VECIR: @GlobVsi = global <4 x i32> zeroinitializer, align 8
// VECIR: @GlobS = global %struct.S zeroinitializer, align 8
// VECIR: %0 = load volatile <4 x i32>, ptr @GlobVsi, align 8
// VECIR: store <4 x i32> %0, ptr getelementptr inbounds (%struct.S, ptr @GlobS, i32 0, i32 1), align 8

// VECASM:      lgrl %r1, GlobVsi@GOT
// VECASM-NEXT: vl   %v0, 0(%r1), 3
// VECASM-NEXT: lgrl %r1, GlobS@GOT
// VECASM-NEXT: vst  %v0, 8(%r1), 3
//
// VECASM:   .globl  GlobVsi
// VECASM:   .p2align        3
// VECASM: GlobVsi:
// VECASM:   .space  16
// VECASM:   .globl  GlobS
// VECASM:   .p2align        3
// VECASM: GlobS:
// VECASM:   .space  24

// SCALIR: %struct.S = type { i32, [12 x i8], <4 x i32> }
// SCALIR: @GlobVsi = global <4 x i32> zeroinitializer, align 16
// SCALIR: @GlobS = global %struct.S zeroinitializer, align 16
// SCALIR: %0 = load volatile <4 x i32>, ptr @GlobVsi, align 16
// SCALIR: store <4 x i32> %0, ptr getelementptr inbounds (%struct.S, ptr @GlobS, i32 0, i32 2), align 16

// SCALASM:      lgrl    %r1, GlobVsi@GOT
// SCALASM-NEXT: l       %r0, 0(%r1)
// SCALASM-NEXT: l       %r2, 4(%r1)
// SCALASM-NEXT: l       %r3, 8(%r1)
// SCALASM-NEXT: l       %r4, 12(%r1)
// SCALASM-NEXT: lgrl    %r1, GlobS@GOT
// SCALASM-NEXT: st      %r4, 28(%r1)
// SCALASM-NEXT: st      %r3, 24(%r1)
// SCALASM-NEXT: st      %r2, 20(%r1)
// SCALASM-NEXT: st      %r0, 16(%r1)
//
// SCALASM:   .globl  GlobVsi
// SCALASM:   .p2align        4
// SCALASM: GlobVsi:
// SCALASM:   .space  16
// SCALASM:   .globl  GlobS
// SCALASM:   .p2align        4
// SCALASM: GlobS:
// SCALASM:   .space  32

