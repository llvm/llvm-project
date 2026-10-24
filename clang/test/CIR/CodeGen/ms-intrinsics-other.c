// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions \
// RUN:   -Wno-implicit-function-declaration -fclangir \
// RUN:   -triple x86_64-unknown-linux -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
//
// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions \
// RUN:   -Wno-implicit-function-declaration -fclangir \
// RUN:   -triple x86_64-unknown-linux -emit-llvm -o %t-cir.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
//
// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions \
// RUN:   -Wno-implicit-function-declaration \
// RUN:   -triple x86_64-unknown-linux -emit-llvm -o %t-ogcg.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-ogcg.ll %s

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
// CIR:         [[ZERO_FWD:%.*]] = cir.const #cir.int<0> : !u32i
// CIR:         [[NONZERO_FWD:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_FWD]] : !u32i
// CIR:         cir.if [[NONZERO_FWD]] {
// CIR:           [[COUNT_FWD:%.*]] = cir.ctz {{%.*}} poison_zero : !u32i
// CIR:           cir.store align(4) [[COUNT_FWD]], {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR:         }
// CIR:         cir.cast bool_to_int [[NONZERO_FWD]] : !cir.bool -> !u8i

// LLVM-LABEL: define{{.*}} @test_BitScanForward(
// LLVM-DAG:     icmp {{(eq|ne)}} i32 %[[MASK_FWD:.*]], 0
// LLVM-DAG:     %[[COUNT_FWD:.*]] = call i32 @llvm.cttz.i32(i32 %[[MASK_FWD]], i1 true)
// LLVM-DAG:     store i32 %[[COUNT_FWD]], ptr %{{.*}}, align 4
// LLVM-DAG:     ret i8

unsigned char test_BitScanReverse(unsigned LONG *Index, unsigned LONG Mask) {
  return _BitScanReverse(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanReverse
// CIR:         [[ZERO_REV:%.*]] = cir.const #cir.int<0> : !u32i
// CIR:         [[NONZERO_REV:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_REV]] : !u32i
// CIR:         cir.if [[NONZERO_REV]] {
// CIR:           [[LAST_REV:%.*]] = cir.const #cir.int<31> : !u32i
// CIR:           [[COUNT_REV:%.*]] = cir.clz {{%.*}} poison_zero : !u32i
// CIR:           [[INDEX_REV:%.*]] = cir.sub nsw [[LAST_REV]], [[COUNT_REV]] : !u32i
// CIR:           cir.store align(4) [[INDEX_REV]], {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR:         }
// CIR:         cir.cast bool_to_int [[NONZERO_REV]] : !cir.bool -> !u8i

// LLVM-LABEL: define{{.*}} @test_BitScanReverse(
// LLVM-DAG:     icmp {{(eq|ne)}} i32 %[[MASK_REV:.*]], 0
// LLVM-DAG:     %[[CLZ_REV:.*]] = call i32 @llvm.ctlz.i32(i32 %[[MASK_REV]], i1 true)
// LLVM-DAG:     %[[INDEX_REV:.*]] = sub nsw i32 31, %[[CLZ_REV]]
// LLVM-DAG:     store i32 %[[INDEX_REV]], ptr %{{.*}}, align 4
// LLVM-DAG:     ret i8

#if defined(__x86_64__)
unsigned char test_BitScanForward64(unsigned LONG *Index, unsigned __int64 Mask) {
  return _BitScanForward64(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanForward64
// CIR:         [[ZERO_FWD64:%.*]] = cir.const #cir.int<0> : !u64i
// CIR:         [[NONZERO_FWD64:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_FWD64]] : !u64i
// CIR:         cir.if [[NONZERO_FWD64]] {
// CIR:           [[COUNT_FWD64:%.*]] = cir.ctz {{%.*}} poison_zero : !u64i
// CIR:           [[INDEX_FWD64:%.*]] = cir.cast integral [[COUNT_FWD64]] : !u64i -> !u32i
// CIR:           cir.store align(4) [[INDEX_FWD64]], {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR:         }
// CIR:         cir.cast bool_to_int [[NONZERO_FWD64]] : !cir.bool -> !u8i

// LLVM-LABEL: define{{.*}} @test_BitScanForward64(
// LLVM-DAG:     icmp {{(eq|ne)}} i64 %[[MASK_FWD64:.*]], 0
// LLVM-DAG:     %[[COUNT_FWD64:.*]] = call i64 @llvm.cttz.i64(i64 %[[MASK_FWD64]], i1 true)
// LLVM-DAG:     %[[TRUNC_FWD64:.*]] = trunc i64 %[[COUNT_FWD64]] to i32
// LLVM-DAG:     store i32 %[[TRUNC_FWD64]], ptr %{{.*}}, align 4
// LLVM-DAG:     ret i8

unsigned char test_BitScanReverse64(unsigned LONG *Index, unsigned __int64 Mask) {
  return _BitScanReverse64(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanReverse64
// CIR:         [[ZERO_REV64:%.*]] = cir.const #cir.int<0> : !u64i
// CIR:         [[NONZERO_REV64:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_REV64]] : !u64i
// CIR:         cir.if [[NONZERO_REV64]] {
// CIR:           [[LAST_REV64:%.*]] = cir.const #cir.int<63> : !u32i
// CIR:           [[COUNT_REV64:%.*]] = cir.clz {{%.*}} poison_zero : !u64i
// CIR:           [[CAST_REV64:%.*]] = cir.cast integral [[COUNT_REV64]] : !u64i -> !u32i
// CIR:           [[INDEX_REV64:%.*]] = cir.sub nsw [[LAST_REV64]], [[CAST_REV64]] : !u32i
// CIR:           cir.store align(4) [[INDEX_REV64]], {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR:         }
// CIR:         cir.cast bool_to_int [[NONZERO_REV64]] : !cir.bool -> !u8i

// LLVM-LABEL: define{{.*}} @test_BitScanReverse64(
// LLVM-DAG:     icmp {{(eq|ne)}} i64 %[[MASK_REV64:.*]], 0
// LLVM-DAG:     %[[CLZ_REV64:.*]] = call i64 @llvm.ctlz.i64(i64 %[[MASK_REV64]], i1 true)
// LLVM-DAG:     %[[TRUNC_REV64:.*]] = trunc i64 %[[CLZ_REV64]] to i32
// LLVM-DAG:     %[[INDEX_REV64:.*]] = sub nsw i32 63, %[[TRUNC_REV64]]
// LLVM-DAG:     store i32 %[[INDEX_REV64]], ptr %{{.*}}, align 4
// LLVM-DAG:     ret i8
#endif

#ifdef __cplusplus
}
#endif
