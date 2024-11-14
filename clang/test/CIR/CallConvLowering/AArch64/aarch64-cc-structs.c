// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-cir-flat -fclangir-call-conv-lowering %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-call-conv-lowering %s -o -| FileCheck %s -check-prefix=LLVM

#include <stdint.h>

typedef struct {
  short a;  
} LT_64;

typedef struct {
  int64_t a;
} EQ_64;

typedef struct {
  int64_t a;
  int b;
} LT_128;

typedef struct {
  int64_t a;
  int64_t b;
} EQ_128;

typedef struct {
  int64_t a;
  int64_t b;
  int64_t c;
} GT_128;

// CHECK: cir.func {{.*@ret_lt_64}}() -> !u16i
// CHECK:   %[[#V0:]] = cir.alloca !ty_LT_64_, !cir.ptr<!ty_LT_64_>, ["__retval"]
// CHECK:   %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_LT_64_>), !cir.ptr<!u16i>
// CHECK:   %[[#V2:]] = cir.load %[[#V1]] : !cir.ptr<!u16i>, !u16i
// CHECK:   cir.return %[[#V2]] : !u16i
LT_64 ret_lt_64() {
  LT_64 x;
  return x;
}

// CHECK: cir.func {{.*@ret_eq_64}}() -> !u64i
// CHECK:   %[[#V0:]] = cir.alloca !ty_EQ_64_, !cir.ptr<!ty_EQ_64_>, ["__retval"]
// CHECK:   %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_EQ_64_>), !cir.ptr<!u64i>
// CHECK:   %[[#V2:]] = cir.load %[[#V1]] : !cir.ptr<!u64i>, !u64i
// CHECK:   cir.return %[[#V2]] : !u64i
EQ_64 ret_eq_64() {
  EQ_64 x;
  return x;
}

// CHECK: cir.func {{.*@ret_lt_128}}() -> !cir.array<!u64i x 2>
// CHECK:   %[[#V0:]] = cir.alloca !ty_LT_128_, !cir.ptr<!ty_LT_128_>, ["__retval"]
// CHECK:   %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_LT_128_>), !cir.ptr<!cir.array<!u64i x 2>>
// CHECK:   %[[#V2:]] = cir.load %[[#V1]] : !cir.ptr<!cir.array<!u64i x 2>>, !cir.array<!u64i x 2>
// CHECK:   cir.return %[[#V2]] : !cir.array<!u64i x 2>
LT_128 ret_lt_128() {
  LT_128 x;
  return x;
}

// CHECK: cir.func {{.*@ret_eq_128}}() -> !cir.array<!u64i x 2>
// CHECK:   %[[#V0:]] = cir.alloca !ty_EQ_128_, !cir.ptr<!ty_EQ_128_>, ["__retval"]
// CHECK:   %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_EQ_128_>), !cir.ptr<!cir.array<!u64i x 2>>
// CHECK:   %[[#V2:]] = cir.load %[[#V1]] : !cir.ptr<!cir.array<!u64i x 2>>, !cir.array<!u64i x 2>
// CHECK:   cir.return %[[#V2]] : !cir.array<!u64i x 2>
EQ_128 ret_eq_128() {
  EQ_128 x;
  return x;
}

// CHECK:     cir.func {{.*@ret_gt_128}}(%arg0: !cir.ptr<!ty_GT_128_> 
// CHECK-NOT:   cir.return {{%.*}}
GT_128 ret_gt_128() {
  GT_128 x;
  return x;
}

typedef struct {
  int a;
  int b;
  int c;  
} S;

// CHECK: cir.func {{.*@retS}}() -> !cir.array<!u64i x 2>
// CHECK:   %[[#V0:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["__retval"] {alignment = 4 : i64}
// CHECK:   %[[#V1:]] = cir.alloca !cir.array<!u64i x 2>, !cir.ptr<!cir.array<!u64i x 2>>, ["tmp"] {alignment = 8 : i64}
// CHECK:   %[[#V2:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S>), !cir.ptr<!void>
// CHECK:   %[[#V3:]] = cir.cast(bitcast, %[[#V1]] : !cir.ptr<!cir.array<!u64i x 2>>), !cir.ptr<!void>
// CHECK:   %[[#V4:]] = cir.const #cir.int<12> : !u64i
// CHECK:   cir.libc.memcpy %[[#V4]] bytes from %[[#V2]] to %[[#V3]] : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
// CHECK:   %[[#V5:]] = cir.load %[[#V1]] : !cir.ptr<!cir.array<!u64i x 2>>, !cir.array<!u64i x 2>
// CHECK:   cir.return %[[#V5]] : !cir.array<!u64i x 2>

// LLVM: [2 x i64] @retS() 
// LLVM:   %[[#V1:]] = alloca %struct.S, i64 1, align 4
// LLVM:   %[[#V2:]] = alloca [2 x i64], i64 1, align 8
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %[[#V2]], ptr %[[#V1]], i64 12, i1 false)
// LLVM:   %[[#V3:]] = load [2 x i64], ptr %[[#V2]], align 8
// LLVM:   ret [2 x i64] %[[#V3]]
S retS() {
  S s;
  return s;
}
// CHECK: cir.func {{.*@pass_lt_64}}(%arg0: !u64
// CHECK:   %[[#V0:]] = cir.alloca !ty_LT_64_, !cir.ptr<!ty_LT_64_>
// CHECK:   %[[#V1:]] = cir.cast(integral, %arg0 : !u64i), !u16i
// CHECK:   %[[#V2:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_LT_64_>), !cir.ptr<!u16i>
// CHECK:   cir.store %[[#V1]], %[[#V2]] : !u16i, !cir.ptr<!u16i>

// LLVM: void @pass_lt_64(i64 %0)
// LLVM:   %[[#V1:]] = alloca %struct.LT_64, i64 1, align 4
// LLVM:   %[[#V2:]] = trunc i64 %0 to i16
// LLVM:   store i16 %[[#V2]], ptr %[[#V1]], align 2
void pass_lt_64(LT_64 s) {}

// CHECK: cir.func {{.*@pass_eq_64}}(%arg0: !u64i
// CHECK:   %[[#V0:]] = cir.alloca !ty_EQ_64_, !cir.ptr<!ty_EQ_64_>
// CHECK:   %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_EQ_64_>), !cir.ptr<!u64i>
// CHECK:   cir.store %arg0, %[[#V1]] : !u64i, !cir.ptr<!u64i>

// LLVM: void @pass_eq_64(i64 %0)
// LLVM:   %[[#V1:]] = alloca %struct.EQ_64, i64 1, align 4
// LLVM:   store i64 %0, ptr %[[#V1]], align 8
void pass_eq_64(EQ_64 s) {}

// CHECK: cir.func {{.*@pass_lt_128}}(%arg0: !cir.array<!u64i x 2>
// CHECK:   %[[#V0:]] = cir.alloca !ty_LT_128_, !cir.ptr<!ty_LT_128_>
// CHECK:   %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_LT_128_>), !cir.ptr<!cir.array<!u64i x 2>>
// CHECK:   cir.store %arg0, %[[#V1]] : !cir.array<!u64i x 2>, !cir.ptr<!cir.array<!u64i x 2>>

// LLVM: void @pass_lt_128([2 x i64] %0)
// LLVM:   %[[#V1:]] = alloca %struct.LT_128, i64 1, align 4
// LLVM:   store [2 x i64] %0, ptr %[[#V1]], align 8
void pass_lt_128(LT_128 s) {}

// CHECK: cir.func {{.*@pass_eq_128}}(%arg0: !cir.array<!u64i x 2>
// CHECK:   %[[#V0:]] = cir.alloca !ty_EQ_128_, !cir.ptr<!ty_EQ_128_>
// CHECK:   %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_EQ_128_>), !cir.ptr<!cir.array<!u64i x 2>>
// CHECK:   cir.store %arg0, %[[#V1]] : !cir.array<!u64i x 2>, !cir.ptr<!cir.array<!u64i x 2>>

// LLVM: void @pass_eq_128([2 x i64] %0)
// LLVM:   %[[#V1]] = alloca %struct.EQ_128, i64 1, align 4
// LLVM:   store [2 x i64] %0, ptr %[[#V1]], align 8
void pass_eq_128(EQ_128 s) {}

// CHECK: cir.func @pass_gt_128(%arg0: !cir.ptr<!ty_GT_128_>
// CHECK:   %[[#V0:]] = cir.alloca !cir.ptr<!ty_GT_128_>, !cir.ptr<!cir.ptr<!ty_GT_128_>>, [""] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %[[#V0]] : !cir.ptr<!ty_GT_128_>, !cir.ptr<!cir.ptr<!ty_GT_128_>>
// CHECK:   %[[#V1:]] = cir.load %[[#V0]] : !cir.ptr<!cir.ptr<!ty_GT_128_>>, !cir.ptr<!ty_GT_128_>

// LLVM: void @pass_gt_128(ptr %0)
// LLVM:   %[[#V1:]] = alloca ptr, i64 1, align 8
// LLVM:   store ptr %0, ptr %[[#V1]], align 8
// LLVM:   %[[#V2:]] = load ptr, ptr %[[#V1]], align 8
void pass_gt_128(GT_128 s) {}

// CHECK: cir.func @get_gt_128(%arg0: !cir.ptr<!ty_GT_128_> {{.*}}, %arg1: !cir.ptr<!ty_GT_128_>
// CHECK: %[[#V0:]] = cir.alloca !cir.ptr<!ty_GT_128_>, !cir.ptr<!cir.ptr<!ty_GT_128_>>, [""] {alignment = 8 : i64}
// CHECK: cir.store %arg1, %[[#V0]] : !cir.ptr<!ty_GT_128_>, !cir.ptr<!cir.ptr<!ty_GT_128_>>
// CHECK: %[[#V1:]] = cir.load %[[#V0]] : !cir.ptr<!cir.ptr<!ty_GT_128_>>, !cir.ptr<!ty_GT_128_>
// CHECK: cir.copy %[[#V1]] to %arg0 : !cir.ptr<!ty_GT_128_>
// CHECK: cir.return

// LLVM: void @get_gt_128(ptr %[[#V0:]], ptr %[[#V1:]])
// LLVM: %[[#V3:]] = alloca ptr, i64 1, align 8
// LLVM: store ptr %[[#V1]], ptr %[[#V3]], align 8
// LLVM: %[[#V4:]] = load ptr, ptr %[[#V3]], align 8
// LLVM: call void @llvm.memcpy.p0.p0.i32(ptr %[[#V0]], ptr %[[#V4]], i32 24, i1 false)
// LLVM: ret void
GT_128 get_gt_128(GT_128 s) {
  return s;
}

// CHECK: cir.func no_proto @call_and_get_gt_128(%arg0: !cir.ptr<!ty_GT_128_>
// CHECK: %[[#V0:]] = cir.alloca !ty_GT_128_, !cir.ptr<!ty_GT_128_>, {{.*}} {alignment = 8 : i64}
// CHECK: %[[#V1:]] = cir.alloca !ty_GT_128_, !cir.ptr<!ty_GT_128_>, {{.*}} {alignment = 8 : i64}
// CHECK: cir.call @get_gt_128(%[[#V1]], %arg0) : (!cir.ptr<!ty_GT_128_>, !cir.ptr<!ty_GT_128_>) -> ()
// CHECK: %[[#V2:]] = cir.load %[[#V1]] : !cir.ptr<!ty_GT_128_>, !ty_GT_128_
// CHECK: cir.store %[[#V2]], %[[#V0]] : !ty_GT_128_, !cir.ptr<!ty_GT_128_>
// CHECK: cir.return

// LLVM: void @call_and_get_gt_128(ptr %[[#V0:]])
// LLVM: %[[#V2:]] = alloca %struct.GT_128, i64 1, align 8
// LLVM: %[[#V3:]] = alloca %struct.GT_128, i64 1, align 8
// LLVM: call void @get_gt_128(ptr %[[#V3]], ptr %[[#V0]])
// LLVM: %[[#V4:]] = load %struct.GT_128, ptr %[[#V3]], align 8
// LLVM: store %struct.GT_128 %[[#V4]], ptr %[[#V2]], align 8
// LLVM: ret void
GT_128 call_and_get_gt_128() {
  GT_128 s;
  s = get_gt_128(s);
  return s;
}
// CHECK: cir.func @passS(%arg0: !cir.array<!u64i x 2> 
// CHECK:   %[[#V0:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, [""] {alignment = 4 : i64}
// CHECK:   %[[#V1:]] = cir.alloca !cir.array<!u64i x 2>, !cir.ptr<!cir.array<!u64i x 2>>, ["tmp"] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %[[#V1]] : !cir.array<!u64i x 2>, !cir.ptr<!cir.array<!u64i x 2>>
// CHECK:   %[[#V2:]] = cir.cast(bitcast, %[[#V1]] : !cir.ptr<!cir.array<!u64i x 2>>), !cir.ptr<!void>
// CHECK:   %[[#V3:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S>), !cir.ptr<!void>
// CHECK:   %[[#V4:]] = cir.const #cir.int<12> : !u64i
// CHECK:   cir.libc.memcpy %[[#V4]] bytes from %[[#V2]] to %[[#V3]] : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>

// LLVM: void @passS([2 x i64] %[[#ARG:]])
// LLVM:   %[[#V1:]] = alloca %struct.S, i64 1, align 4
// LLVM:   %[[#V2:]] = alloca [2 x i64], i64 1, align 8
// LLVM:   store [2 x i64] %[[#ARG]], ptr %[[#V2]], align 8
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %[[#V1]], ptr %[[#V2]], i64 12, i1 false)
void passS(S s) {}