// RUN: %clang_cc1 %s -triple aarch64-none-linux-android21 -fclangir -emit-cir -std=c11 -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 %s -triple aarch64-none-linux-android21 -fclangir -emit-llvm -std=c11 -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// CIR-DAG: ![[PS:.*]] = !cir.record<struct "PS" {!s16i, !s16i, !s16i}
// CIR-DAG: ![[ANON:.*]] = !cir.record<struct  {![[PS]], !cir.array<!u8i x 2>}>
// CIR-DAG: cir.global external @testPromotedStructGlobal = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s16i, #cir.int<2> : !s16i, #cir.int<3> : !s16i}> : ![[PS]], #cir.zero : !cir.array<!u8i x 2>}> : ![[ANON]]

// LLVM-DAG: %[[PS:.*]] = type { i16, i16, i16 }
// LLVM-DAG: @testPromotedStructGlobal = global { %[[PS]], [2 x i8] } { %[[PS]] { i16 1, i16 2, i16 3 }, [2 x i8] zeroinitializer }
typedef struct { short x, y, z; } PS;
_Atomic PS testPromotedStructGlobal = (PS){1, 2, 3};
