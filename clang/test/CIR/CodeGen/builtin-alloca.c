// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

typedef __SIZE_TYPE__ size_t;
void *alloca(size_t size);
void *_alloca(size_t size);

void my_alloca(size_t n)
{
  int *c1 = alloca(n);
}
// CIR:       cir.func @my_alloca([[ALLOCA_SIZE:%.*]]: !u64i
// CIR:       cir.store [[ALLOCA_SIZE]], [[LOCAL_VAR_ALLOCA_SIZE:%.*]] : !u64i, cir.ptr <!u64i>
// CIR:       [[TMP_ALLOCA_SIZE:%.*]] = cir.load [[LOCAL_VAR_ALLOCA_SIZE]] : cir.ptr <!u64i>, !u64i
// CIR:       [[ALLOCA_RES:%.*]] = cir.alloca !u8i, cir.ptr <!u8i>, [[TMP_ALLOCA_SIZE]] : !u64i, ["bi_alloca"] {alignment = 16 : i64}
// CIR-NEXT:  cir.cast(bitcast, [[ALLOCA_RES]] : !cir.ptr<!u8i>), !cir.ptr<!void>
// CIR: }


// LLVM:       define void @my_alloca(i64 [[ALLOCA_SIZE:%.*]])
// LLVM:       store i64 [[ALLOCA_SIZE]], ptr [[LOCAL_VAR_ALLOCA_SIZE:%.*]],
// LLVM:       [[TMP_ALLOCA_SIZE:%.*]] =  load i64, ptr [[LOCAL_VAR_ALLOCA_SIZE]],
// LLVM:       [[ALLOCA_RES:%.*]] = alloca i8, i64 [[TMP_ALLOCA_SIZE]], align 16
// LLVM: }

void my___builtin_alloca(size_t n)
{
  int *c1 = (int *)__builtin_alloca(n);
}

// CIR:       cir.func @my___builtin_alloca([[ALLOCA_SIZE:%.*]]: !u64i
// CIR:       cir.store [[ALLOCA_SIZE]], [[LOCAL_VAR_ALLOCA_SIZE:%.*]] : !u64i, cir.ptr <!u64i>
// CIR:       [[TMP_ALLOCA_SIZE:%.*]] = cir.load [[LOCAL_VAR_ALLOCA_SIZE]] : cir.ptr <!u64i>, !u64i
// CIR:       [[ALLOCA_RES:%.*]] = cir.alloca !u8i, cir.ptr <!u8i>, [[TMP_ALLOCA_SIZE]] : !u64i, ["bi_alloca"] {alignment = 16 : i64}
// CIR-NEXT:  cir.cast(bitcast, [[ALLOCA_RES]] : !cir.ptr<!u8i>), !cir.ptr<!void>
// CIR: }


// LLVM:       define void @my___builtin_alloca(i64 [[ALLOCA_SIZE:%.*]])
// LLVM:       store i64 [[ALLOCA_SIZE]], ptr [[LOCAL_VAR_ALLOCA_SIZE:%.*]],
// LLVM:       [[TMP_ALLOCA_SIZE:%.*]] =  load i64, ptr [[LOCAL_VAR_ALLOCA_SIZE]],
// LLVM:       [[ALLOCA_RES:%.*]] = alloca i8, i64 [[TMP_ALLOCA_SIZE]], align 16
// LLVM: }

void my__builtin_alloca_uninitialized(size_t n)
{
  int *c1 = (int *)__builtin_alloca_uninitialized(n);
}

// CIR:       cir.func @my__builtin_alloca_uninitialized([[ALLOCA_SIZE:%.*]]: !u64i
// CIR:       cir.store [[ALLOCA_SIZE]], [[LOCAL_VAR_ALLOCA_SIZE:%.*]] : !u64i, cir.ptr <!u64i>
// CIR:       [[TMP_ALLOCA_SIZE:%.*]] = cir.load [[LOCAL_VAR_ALLOCA_SIZE]] : cir.ptr <!u64i>, !u64i
// CIR:       [[ALLOCA_RES:%.*]] = cir.alloca !u8i, cir.ptr <!u8i>, [[TMP_ALLOCA_SIZE]] : !u64i, ["bi_alloca"] {alignment = 16 : i64}
// CIR-NEXT:  cir.cast(bitcast, [[ALLOCA_RES]] : !cir.ptr<!u8i>), !cir.ptr<!void>
// CIR: }


// LLVM:       define void @my__builtin_alloca_uninitialized(i64 [[ALLOCA_SIZE:%.*]])
// LLVM:       store i64 [[ALLOCA_SIZE]], ptr [[LOCAL_VAR_ALLOCA_SIZE:%.*]],
// LLVM:       [[TMP_ALLOCA_SIZE:%.*]] =  load i64, ptr [[LOCAL_VAR_ALLOCA_SIZE]],
// LLVM:       [[ALLOCA_RES:%.*]] = alloca i8, i64 [[TMP_ALLOCA_SIZE]], align 16
// LLVM: }
