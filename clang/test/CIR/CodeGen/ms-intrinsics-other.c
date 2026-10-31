// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions \
// RUN:   -Wno-implicit-function-declaration -fclangir \
// RUN:   -triple x86_64-unknown-linux -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
//
// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions \
// RUN:   -Wno-implicit-function-declaration -fclangir \
// RUN:   -triple x86_64-unknown-linux -emit-llvm -o %t-cir.ll %s
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
//
// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions \
// RUN:   -Wno-implicit-function-declaration \
// RUN:   -triple x86_64-unknown-linux -emit-llvm -o %t-ogcg.ll %s
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t-ogcg.ll %s

#ifdef __LP64__
#define LONG int
#else
#define LONG long
#endif

#ifdef __cplusplus
extern "C" {
#endif

unsigned char test_BitScanForward(unsigned LONG *Index, unsigned LONG Mask) {
  return _BitScanForward(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanForward
// CIR:       [[ZERO_FWD:%.*]] = cir.const #cir.int<0> : !u32i
// CIR:       [[NONZERO_FWD:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_FWD]] : !u32i
// CIR:       cir.if [[NONZERO_FWD]] {
// CIR:         [[COUNT_FWD:%.*]] = cir.ctz {{%.*}} poison_zero : !u32i
// CIR:         cir.store align(4) [[COUNT_FWD]], {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR:       }
// CIR:       cir.cast bool_to_int [[NONZERO_FWD]] : !cir.bool -> !u8i

// The CIR pipeline branches to the bit-scan block when Mask != 0.
// OGCG branches to the end block when Mask == 0. The condition and
// destinations are reversed, but both paths execute the bit scan only
// for a nonzero Mask and then rejoin the end block.

// LLVM-LABEL: define{{.*}} @test_BitScanForward(
// LLVMCIR:       [[COND_FWD:%.*]] = icmp ne i32 [[MASK_FWD:%.*]], 0
// LLVMCIR:   br i1 [[COND_FWD]], label %[[NOT_ZERO_FWD:.*]], label %[[END_FWD:.*]]
// LLVMCIR:       [[NOT_ZERO_FWD]]:
// LLVMCIR:   [[COUNT_FWD:%.*]] = call i32 @llvm.cttz.i32(i32 [[MASK_FWD]], i1 true)
// LLVMCIR:   store i32 [[COUNT_FWD]], ptr [[INDEX_FWD:%.*]], align 4
// LLVMCIR:   br label %[[END_FWD]]
// LLVMCIR:       [[END_FWD]]:
// LLVMCIR:         zext i1 [[COND_FWD]] to i8
// LLVMCIR:         ret i8
//
// OGCG:         [[COND_FWD:%.*]] = icmp eq i32 [[MASK_FWD:%.*]], 0
// OGCG:     br i1 [[COND_FWD]], label %[[END_FWD:.*]], label %[[NOT_ZERO_FWD:.*]]
// OGCG:       [[END_FWD]]:
// OGCG:     phi i8 [ 0, %{{.*}} ], [ 1, %[[NOT_ZERO_FWD]] ]
// OGCG:     ret i8
// OGCG:       [[NOT_ZERO_FWD]]:
// OGCG:     [[COUNT_FWD:%.*]] = call i32 @llvm.cttz.i32(i32 [[MASK_FWD]], i1 true)
// OGCG:     store i32 [[COUNT_FWD]], ptr [[INDEX_FWD:%.*]], align 4
// OGCG:     br label %[[END_FWD]]

unsigned char test_BitScanReverse(unsigned LONG *Index, unsigned LONG Mask) {
  return _BitScanReverse(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanReverse
// CIR:       [[ZERO_REV:%.*]] = cir.const #cir.int<0> : !u32i
// CIR:       [[NONZERO_REV:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_REV]] : !u32i
// CIR:       cir.if [[NONZERO_REV]] {
// CIR:         [[LAST_REV:%.*]] = cir.const #cir.int<31> : !u32i
// CIR:         [[COUNT_REV:%.*]] = cir.clz {{%.*}} poison_zero : !u32i
// CIR:         [[INDEX_REV:%.*]] = cir.sub nsw [[LAST_REV]], [[COUNT_REV]] : !u32i
// CIR:         cir.store align(4) [[INDEX_REV]], {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR:       }
// CIR:       cir.cast bool_to_int [[NONZERO_REV]] : !cir.bool -> !u8i

// LLVM-LABEL: define{{.*}} @test_BitScanReverse(
// LLVMCIR:       [[COND_REV:%.*]] = icmp ne i32 [[MASK_REV:%.*]], 0
// LLVMCIR:   br i1 [[COND_REV]], label %[[NOT_ZERO_REV:.*]], label %[[END_REV:.*]]
// LLVMCIR:       [[NOT_ZERO_REV]]:
// LLVMCIR:   [[CLZ_REV:%.*]] = call i32 @llvm.ctlz.i32(i32 [[MASK_REV]], i1 true)
// LLVMCIR:   [[INDEX_REV:%.*]] = sub nsw i32 31, [[CLZ_REV]]
// LLVMCIR:   store i32 [[INDEX_REV]], ptr [[INDEX_PTR_REV:%.*]], align 4
// LLVMCIR:   br label %[[END_REV]]
// LLVMCIR:       [[END_REV]]:
// LLVMCIR:         zext i1 [[COND_REV]] to i8
// LLVMCIR:         ret i8
//
// OGCG:         [[COND_REV:%.*]] = icmp eq i32 [[MASK_REV:%.*]], 0
// OGCG:     br i1 [[COND_REV]], label %[[END_REV:.*]], label %[[NOT_ZERO_REV:.*]]
// OGCG:       [[END_REV]]:
// OGCG:     phi i8 [ 0, %{{.*}} ], [ 1, %[[NOT_ZERO_REV]] ]
// OGCG:     ret i8
// OGCG:       [[NOT_ZERO_REV]]:
// OGCG:     [[CLZ_REV:%.*]] = call i32 @llvm.ctlz.i32(i32 [[MASK_REV]], i1 true)
// OGCG:     [[INDEX_REV:%.*]] = sub nsw i32 31, [[CLZ_REV]]
// OGCG:     store i32 [[INDEX_REV]], ptr [[INDEX_PTR_REV:%.*]], align 4
// OGCG:     br label %[[END_REV]]

#if defined(__x86_64__)
unsigned char test_BitScanForward64(unsigned LONG *Index, unsigned __int64 Mask) {
  return _BitScanForward64(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanForward64
// CIR:       [[ZERO_FWD64:%.*]] = cir.const #cir.int<0> : !u64i
// CIR:       [[NONZERO_FWD64:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_FWD64]] : !u64i
// CIR:       cir.if [[NONZERO_FWD64]] {
// CIR:         [[COUNT_FWD64:%.*]] = cir.ctz {{%.*}} poison_zero : !u64i
// CIR:         [[INDEX_FWD64:%.*]] = cir.cast integral [[COUNT_FWD64]] : !u64i -> !u32i
// CIR:         cir.store align(4) [[INDEX_FWD64]], {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR:       }
// CIR:       cir.cast bool_to_int [[NONZERO_FWD64]] : !cir.bool -> !u8i

// LLVM-LABEL: define{{.*}} @test_BitScanForward64(
// LLVMCIR:       [[COND_FWD64:%.*]] = icmp ne i64 [[MASK_FWD64:%.*]], 0
// LLVMCIR:   br i1 [[COND_FWD64]], label %[[NOT_ZERO_FWD64:.*]], label %[[END_FWD64:.*]]
// LLVMCIR:       [[NOT_ZERO_FWD64]]:
// LLVMCIR:   [[COUNT_FWD64:%.*]] = call i64 @llvm.cttz.i64(i64 [[MASK_FWD64]], i1 true)
// LLVMCIR:   [[TRUNC_FWD64:%.*]] = trunc i64 [[COUNT_FWD64]] to i32
// LLVMCIR:   store i32 [[TRUNC_FWD64]], ptr [[INDEX_FWD64:%.*]], align 4
// LLVMCIR:   br label %[[END_FWD64]]
// LLVMCIR:       [[END_FWD64]]:
// LLVMCIR:         zext i1 [[COND_FWD64]] to i8
// LLVMCIR:         ret i8
//
// OGCG:         [[COND_FWD64:%.*]] = icmp eq i64 [[MASK_FWD64:%.*]], 0
// OGCG:     br i1 [[COND_FWD64]], label %[[END_FWD64:.*]], label %[[NOT_ZERO_FWD64:.*]]
// OGCG:       [[END_FWD64]]:
// OGCG:     phi i8 [ 0, %{{.*}} ], [ 1, %[[NOT_ZERO_FWD64]] ]
// OGCG:     ret i8
// OGCG:       [[NOT_ZERO_FWD64]]:
// OGCG:     [[COUNT_FWD64:%.*]] = call i64 @llvm.cttz.i64(i64 [[MASK_FWD64]], i1 true)
// OGCG:     [[TRUNC_FWD64:%.*]] = trunc i64 [[COUNT_FWD64]] to i32
// OGCG:     store i32 [[TRUNC_FWD64]], ptr [[INDEX_PTR_FWD64:%.*]], align 4
// OGCG:     br label %[[END_FWD64]]

unsigned char test_BitScanReverse64(unsigned LONG *Index, unsigned __int64 Mask) {
  return _BitScanReverse64(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanReverse64
// CIR:       [[ZERO_REV64:%.*]] = cir.const #cir.int<0> : !u64i
// CIR:       [[NONZERO_REV64:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_REV64]] : !u64i
// CIR:       cir.if [[NONZERO_REV64]] {
// CIR:         [[LAST_REV64:%.*]] = cir.const #cir.int<63> : !u32i
// CIR:         [[COUNT_REV64:%.*]] = cir.clz {{%.*}} poison_zero : !u64i
// CIR:         [[CAST_REV64:%.*]] = cir.cast integral [[COUNT_REV64]] : !u64i -> !u32i
// CIR:         [[INDEX_REV64:%.*]] = cir.sub nsw [[LAST_REV64]], [[CAST_REV64]] : !u32i
// CIR:         cir.store align(4) [[INDEX_REV64]], {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR:       }
// CIR:       cir.cast bool_to_int [[NONZERO_REV64]] : !cir.bool -> !u8i

// LLVM-LABEL: define{{.*}} @test_BitScanReverse64(
// LLVMCIR:       [[COND_REV64:%.*]] = icmp ne i64 [[MASK_REV64:%.*]], 0
// LLVMCIR:   br i1 [[COND_REV64]], label %[[NOT_ZERO_REV64:.*]], label %[[END_REV64:.*]]
// LLVMCIR:       [[NOT_ZERO_REV64]]:
// LLVMCIR:   [[CLZ_REV64:%.*]] = call i64 @llvm.ctlz.i64(i64 [[MASK_REV64]], i1 true)
// LLVMCIR:   [[TRUNC_REV64:%.*]] = trunc i64 [[CLZ_REV64]] to i32
// LLVMCIR:   [[INDEX_REV64:%.*]] = sub nsw i32 63, [[TRUNC_REV64]]
// LLVMCIR:   store i32 [[INDEX_REV64]], ptr [[INDEX_PTR_REV64:%.*]], align 4
// LLVMCIR:   br label %[[END_REV64]]
// LLVMCIR:       [[END_REV64]]:
// LLVMCIR:         zext i1 [[COND_REV64]] to i8
// LLVMCIR:         ret i8
//
// OGCG:         [[COND_REV64:%.*]] = icmp eq i64 [[MASK_REV64:%.*]], 0
// OGCG:     br i1 [[COND_REV64]], label %[[END_REV64:.*]], label %[[NOT_ZERO_REV64:.*]]
// OGCG:       [[END_REV64]]:
// OGCG:     phi i8 [ 0, %{{.*}} ], [ 1, %[[NOT_ZERO_REV64]] ]
// OGCG:     ret i8
// OGCG:       [[NOT_ZERO_REV64]]:
// OGCG:     [[CLZ_REV64:%.*]] = call i64 @llvm.ctlz.i64(i64 [[MASK_REV64]], i1 true)
// OGCG:     [[TRUNC_REV64:%.*]] = trunc i64 [[CLZ_REV64]] to i32
// OGCG:     [[INDEX_REV64:%.*]] = sub nsw i32 63, [[TRUNC_REV64]]
// OGCG:     store i32 [[INDEX_REV64]], ptr [[INDEX_PTR_REV64:.*]], align 4
// OGCG:     br label %[[END_REV64]]

#endif

#ifdef __cplusplus
}
#endif
