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
