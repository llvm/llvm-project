// RUN: %clang_cc1 -ffreestanding -fms-extensions -triple x86_64-unknown-linux-gnu \
// RUN:         -Oz -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

// RUN: %clang_cc1 -ffreestanding -fms-extensions -triple x86_64-unknown-linux-gnu \
// RUN:         -Oz -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// RUN: %clang_cc1 -ffreestanding -fms-extensions -triple x86_64-unknown-linux-gnu \
// RUN:         -Oz -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

unsigned __int64 __shiftleft128(unsigned __int64 low, unsigned __int64 high,
                                unsigned char shift);
unsigned __int64 __shiftright128(unsigned __int64 low, unsigned __int64 high,
                                 unsigned char shift);

// CIR-LABEL: cir.func{{.*}}@test_shiftleft128
// CIR: %[[D_LOAD:[^ ]+]] = cir.load {{.*}} : !cir.ptr<!u8i>, !u8i
// CIR: %[[D_CAST:[^ ]+]] = cir.cast integral %[[D_LOAD]] : !u8i -> !u64i
// CIR: %{{[^ ]+}} = cir.call_llvm_intrinsic "fshl" {{.*}} : (!u64i, !u64i, !u64i) -> !u64i
// CIR: cir.return

// LLVM-LABEL: define {{.*}} i64 @test_shiftleft128
// LLVM-SAME: (i64 %[[ARG0:.*]], i64 %[[ARG1:.*]], i8 %[[ARG2:.*]])
// LLVM-NEXT: [[TMP1:%.*]] = zext i8 %[[ARG2]] to i64
// LLVM-NEXT: [[TMP2:%.*]] = tail call i64 @llvm.fshl.i64(i64 %[[ARG1]], i64 %[[ARG0]], i64 [[TMP1]])

// OGCG-LABEL: define {{.*}} i64 @test_shiftleft128
// OGCG-SAME: (i64 {{.*}} %[[ARG0:.*]], i64 {{.*}} %[[ARG1:.*]], i8 {{.*}} %[[ARG2:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT: [[TMP0:%.*]] = zext i8 %[[ARG2]] to i64
// OGCG-NEXT: [[TMP1:%.*]] = tail call i64 @llvm.fshl.i64(i64 %[[ARG1]], i64 %[[ARG0]], i64 [[TMP0]])
// OGCG-NEXT: ret i64 [[TMP1]]
unsigned __int64 test_shiftleft128(unsigned __int64 l, unsigned __int64 h,
                                   unsigned char d) {
  return __shiftleft128(l, h, d);
}

// CIR-LABEL: cir.func{{.*}}@test_shiftright128
// CIR: %[[D_LOAD:[^ ]+]] = cir.load {{.*}} : !cir.ptr<!u8i>, !u8i
// CIR: %[[D_CAST:[^ ]+]] = cir.cast integral %[[D_LOAD]] : !u8i -> !u64i
// CIR: %{{[^ ]+}} = cir.call_llvm_intrinsic "fshr" {{.*}} : (!u64i, !u64i, !u64i) -> !u64i
// CIR: cir.return

// LLVM-LABEL: define {{.*}} i64 @test_shiftright128
// LLVM-SAME: (i64 %[[ARG0:.*]], i64 %[[ARG1:.*]], i8 %[[ARG2:.*]])
// LLVM-NEXT: [[TMP1:%.*]] = zext i8 %[[ARG2]] to i64
// LLVM-NEXT: [[TMP2:%.*]] = tail call i64 @llvm.fshr.i64(i64 %[[ARG1]], i64 %[[ARG0]], i64 [[TMP1]])

// OGCG-LABEL: define {{.*}} i64 @test_shiftright128
// OGCG-SAME: (i64 {{.*}} %[[ARG0:.*]], i64 {{.*}} %[[ARG1:.*]], i8 {{.*}} %[[ARG2:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT: [[TMP0:%.*]] = zext i8 %[[ARG2]] to i64
// OGCG-NEXT: [[TMP1:%.*]] = tail call i64 @llvm.fshr.i64(i64 %[[ARG1]], i64 %[[ARG0]], i64 [[TMP0]])
// OGCG-NEXT: ret i64 [[TMP1]]
unsigned __int64 test_shiftright128(unsigned __int64 l, unsigned __int64 h,
                                    unsigned char d) {
  return __shiftright128(l, h, d);
}

#pragma intrinsic(__cpuid)
#pragma intrinsic(__cpuidex)

void test__cpuid_with_cpu_info_as_pointer(int cpuInfo[4], int functionId) {
    __cpuid(cpuInfo, functionId);

    // CIR-LABEL: __cpuid_with_cpu_info_as_pointer
    // CIR: [[CPU_INFO_PTR:%.*]] = cir.load align(8)
    // CIR: [[FUNCTION_ID:%.*]] = cir.load align(4)
    // CIR: [[SUB_FUNCTION_ID:%.*]] = cir.const #cir.int<0> : !s32i
    // CIR: cir.cpuid [[CPU_INFO_PTR]], [[FUNCTION_ID]], [[SUB_FUNCTION_ID]] : !cir.ptr<!s32i>, !s32i, !s32i

    // LLVM-LABEL: __cpuid_with_cpu_info_as_pointer
    // LLVM: [[CPU_INFO_PTR:%.*]] = load ptr
    // LLVM: [[FUNCTION_ID:%.*]] = load i32
    // LLVM: [[ASM_RESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 [[FUNCTION_ID]], i32 0)
    // LLVM: [[RESULT_0:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 0
    // LLVM: [[ADDR_PTR_0:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 0
    // LLVM: store i32 [[RESULT_0]], ptr [[ADDR_PTR_0]], align 4
    // LLVM: [[RESULT_1:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 1
    // LLVM: [[ADDR_PTR_1:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 1
    // LLVM: store i32 [[RESULT_1]], ptr [[ADDR_PTR_1]], align 4
    // LLVM: [[RESULT_2:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 2
    // LLVM: [[ADDR_PTR_2:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 2
    // LLVM: store i32 [[RESULT_2]], ptr [[ADDR_PTR_2]], align 4
    // LLVM: [[RESULT_3:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 3
    // LLVM: [[ADDR_PTR_3:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 3
    // LLVM: store i32 [[RESULT_3]], ptr [[ADDR_PTR_3]], align 4

    // OGCG-LABEL: __cpuid_with_cpu_info_as_pointer
    // OGCG: [[FUNCTION_ID:%.*]] = load i32
    // OGCG: [[FUNCTION_ID_ARG:%.*]] = load i32
    // OGCG: [[ASM_RESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 [[FUNCTION_ID_ARG]], i32 0)
    // OGCG: [[CPU_INFO_PTR:%.*]] = load ptr
    // OGCG: [[RESULT_0:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 0
    // OGCG: [[ADDR_PTR_0:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 0
    // OGCG: store i32 [[RESULT_0]], ptr [[ADDR_PTR_0]], align 4
    // OGCG: [[RESULT_1:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 1
    // OGCG: [[ADDR_PTR_1:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 1
    // OGCG: store i32 [[RESULT_1]], ptr [[ADDR_PTR_1]], align 4
    // OGCG: [[RESULT_2:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 2
    // OGCG: [[ADDR_PTR_2:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 2
    // OGCG: store i32 [[RESULT_2]], ptr [[ADDR_PTR_2]], align 4
    // OGCG: [[RESULT_3:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 3
    // OGCG: [[ADDR_PTR_3:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 3
    // OGCG: store i32 [[RESULT_3]], ptr [[ADDR_PTR_3]], align 4
}

void test__cpuid_with_cpu_info_as_array(int functionId) {
    int cpuInfo[4];
    __cpuid(cpuInfo, functionId);

    // CIR-LABEL: _cpuid_with_cpu_info_as_array
    // CIR: [[CPU_INFO:%.*]] = cir.cast array_to_ptrdecay
    // CIR: [[FUNCTION_ID:%.*]] = cir.load align(4)
    // CIR: [[SUB_FUNCTION_ID:%.*]] = cir.const #cir.int<0> : !s32i
    // CIR: cir.cpuid [[CPU_INFO_PTR]], [[FUNCTION_ID]], [[SUB_FUNCTION_ID]] : !cir.ptr<!s32i>, !s32i, !s32i

    // LLVM-LABEL: _cpuid_with_cpu_info_as_array
    // LLVM: [[CPU_INFO_PTR:%.*]] = getelementptr i32, ptr
    // LLVM: [[FUNCTION_ID:%.*]] = load i32
    // LLVM: [[ASM_RESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 [[FUNCTION_ID]], i32 0)
    // LLVM: [[RESULT_0:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 0
    // LLVM: [[ADDR_PTR_0:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 0
    // LLVM: store i32 [[RESULT_0]], ptr [[ADDR_PTR_0]], align 4
    // LLVM: [[RESULT_1:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 1
    // LLVM: [[ADDR_PTR_1:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 1
    // LLVM: store i32 [[RESULT_1]], ptr [[ADDR_PTR_1]], align 4
    // LLVM: [[RESULT_2:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 2
    // LLVM: [[ADDR_PTR_2:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 2
    // LLVM: store i32 [[RESULT_2]], ptr [[ADDR_PTR_2]], align 4
    // LLVM: [[RESULT_3:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 3
    // LLVM: [[ADDR_PTR_3:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 3
    // LLVM: store i32 [[RESULT_3]], ptr [[ADDR_PTR_3]], align 4

    // OGCG-LABEL: _cpuid_with_cpu_info_as_array
    // OGCG: [[FUNCTION_ID:%.*]] = load i32
    // OGCG: [[FUNCTION_ID_ARG:%.*]] = load i32
    // OGCG: [[ASM_RESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 [[FUNCTION_ID_ARG]], i32 0)
    // OGCG: [[CPU_INFO_PTR:%.*]] = getelementptr inbounds [4 x i32], ptr
    // OGCG: [[RESULT_0:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 0
    // OGCG: [[ADDR_PTR_0:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 0
    // OGCG: store i32 [[RESULT_0]], ptr [[ADDR_PTR_0]], align 4
    // OGCG: [[RESULT_1:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 1
    // OGCG: [[ADDR_PTR_1:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 1
    // OGCG: store i32 [[RESULT_1]], ptr [[ADDR_PTR_1]], align 4
    // OGCG: [[RESULT_2:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 2
    // OGCG: [[ADDR_PTR_2:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 2
    // OGCG: store i32 [[RESULT_2]], ptr [[ADDR_PTR_2]], align 4
    // OGCG: [[RESULT_3:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 3
    // OGCG: [[ADDR_PTR_3:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 3
    // OGCG: store i32 [[RESULT_3]], ptr [[ADDR_PTR_3]], align 4
}
void test__cpuidex(int cpuInfo[4], int functionId, int subFunctionId) {
    __cpuidex(cpuInfo, functionId, subFunctionId);

    // CIR-LABEL: __cpuidex
    // CIR: [[CPU_INFO_PTR:%.*]] = cir.load align(8)
    // CIR: [[FUNCTION_ID:%.*]] = cir.load align(4)
    // CIR: [[SUB_FUNCTION_ID:%.*]] = cir.load align(4)
    // CIR: cir.cpuid [[CPU_INFO_PTR]], [[FUNCTION_ID]], [[SUB_FUNCTION_ID]] : !cir.ptr<!s32i>, !s32i, !s32i

    // LLVM-LABEL: __cpuidex
    // LLVM: [[CPU_INFO_PTR:%.*]] = load ptr
    // LLVM: [[FUNCTION_ID:%.*]] = load i32
    // LLVM: [[SUB_FUNCTION_ID:%.*]] = load i32
    // LLVM: [[ASM_RESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 [[FUNCTION_ID]], i32 [[SUB_FUNCTION_ID]])
    // LLVM: [[RESULT_0:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 0
    // LLVM: [[ADDR_PTR_0:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 0
    // LLVM: store i32 [[RESULT_0]], ptr [[ADDR_PTR_0]], align 4
    // LLVM: [[RESULT_1:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 1
    // LLVM: [[ADDR_PTR_1:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 1
    // LLVM: store i32 [[RESULT_1]], ptr [[ADDR_PTR_1]], align 4
    // LLVM: [[RESULT_2:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 2
    // LLVM: [[ADDR_PTR_2:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 2
    // LLVM: store i32 [[RESULT_2]], ptr [[ADDR_PTR_2]], align 4
    // LLVM: [[RESULT_3:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 3
    // LLVM: [[ADDR_PTR_3:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 3
    // LLVM: store i32 [[RESULT_3]], ptr [[ADDR_PTR_3]], align 4

    // OGCG-LABEL: __cpuidex
    // OGCG: [[FUNCTION_ID:%.*]] = load i32
    // OGCG: [[SUB_FUNCTION_ID:%.*]] = load i32
    // OGCG: [[FUNCTION_ID_ARG:%.*]] = load i32
    // OGCG: [[SUB_FUNCTION_ID_ARG:%.*]] = load i32
    // OGCG: [[ASM_RESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 [[FUNCTION_ID_ARG]], i32 [[SUB_FUNCTION_ID_ARG]])
    // OGCG: [[CPU_INFO_PTR:%.*]] = load ptr
    // OGCG: [[RESULT_0:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 0
    // OGCG: [[ADDR_PTR_0:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 0
    // OGCG: store i32 [[RESULT_0]], ptr [[ADDR_PTR_0]], align 4
    // OGCG: [[RESULT_1:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 1
    // OGCG: [[ADDR_PTR_1:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 1
    // OGCG: store i32 [[RESULT_1]], ptr [[ADDR_PTR_1]], align 4
    // OGCG: [[RESULT_2:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 2
    // OGCG: [[ADDR_PTR_2:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 2
    // OGCG: store i32 [[RESULT_2]], ptr [[ADDR_PTR_2]], align 4
    // OGCG: [[RESULT_3:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASM_RESULTS]], 3
    // OGCG: [[ADDR_PTR_3:%.*]] = getelementptr inbounds i32, ptr [[CPU_INFO_PTR]], i32 3
    // OGCG: store i32 [[RESULT_3]], ptr [[ADDR_PTR_3]], align 4
>>>>>>> 8b09e47e6fc8 ([CIR][X86] Add support for `cpuid`/`cpuidex` (#173197))
}
