// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -fms-extensions -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir -Wall -Werror %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -fms-extensions -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t-cir.ll -Wall -Werror %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -fms-extensions -Wno-implicit-function-declaration -emit-llvm -o %t.ll -Wall -Werror %s
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

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
}
