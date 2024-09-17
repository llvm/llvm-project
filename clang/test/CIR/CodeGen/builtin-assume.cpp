// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir

int test_assume(int x) {
  __builtin_assume(x > 0);
  return x;
}

//      CIR: cir.func @_Z11test_assumei
//      CIR:   %[[#x:]] = cir.load %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#zero:]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:   %[[#cond:]] = cir.cmp(gt, %[[#x]], %[[#zero]]) : !s32i, !cir.bool
// CIR-NEXT:   cir.assume %[[#cond]] : !cir.bool
//      CIR: }

int test_assume_aligned(int *ptr) {
  int *aligned = (int *)__builtin_assume_aligned(ptr, 8);
  return *aligned;
}

//      CIR: cir.func @_Z19test_assume_alignedPi
//      CIR:   %[[#ptr:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[#aligned:]] = cir.assume.aligned %[[#ptr]] : !cir.ptr<!s32i>[alignment 8]
// CIR-NEXT:   cir.store %[[#aligned]], %[[#aligned_slot:]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:   %[[#aligned2:]] = cir.load deref %[[#aligned_slot]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %{{.+}} = cir.load %[[#aligned2]] : !cir.ptr<!s32i>, !s32i
//      CIR: }

int test_assume_aligned_offset(int *ptr) {
  int *aligned = (int *)__builtin_assume_aligned(ptr, 8, 4);
  return *aligned;
}

//      CIR: cir.func @_Z26test_assume_aligned_offsetPi
//      CIR:   %[[#ptr:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[#offset:]] = cir.const #cir.int<4> : !s32i
// CIR-NEXT:   %[[#offset2:]] = cir.cast(integral, %[[#offset]] : !s32i), !u64i
// CIR-NEXT:   %[[#aligned:]] = cir.assume.aligned %[[#ptr]] : !cir.ptr<!s32i>[alignment 8, offset %[[#offset2]] : !u64i]
// CIR-NEXT:   cir.store %[[#aligned]], %[[#aligned_slot:]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:   %[[#aligned2:]] = cir.load deref %[[#aligned_slot]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %{{.+}} = cir.load %[[#aligned2]] : !cir.ptr<!s32i>, !s32i
//      CIR: }

int test_separate_storage(int *p1, int *p2) {
  __builtin_assume_separate_storage(p1, p2);
  return *p1 + *p2;
}

//      CIR: cir.func @_Z21test_separate_storagePiS_
//      CIR:   %[[#p1:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[#p1_voidptr:]] = cir.cast(bitcast, %[[#p1]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CIR-NEXT:   %[[#p2:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[#p2_voidptr:]] = cir.cast(bitcast, %[[#p2]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CIR-NEXT:   cir.assume.separate_storage %[[#p1_voidptr]], %[[#p2_voidptr]] : !cir.ptr<!void>
//      CIR: }
