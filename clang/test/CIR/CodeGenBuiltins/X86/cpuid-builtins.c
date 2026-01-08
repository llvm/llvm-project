// RUN: %clang_cc1 -triple x86_64-unknown-linux -fms-extensions -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir -Wall -Werror %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fms-extensions -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t-cir.ll -Wall -Werror %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fms-extensions -Wno-implicit-function-declaration -emit-llvm -o %t.ll -Wall -Werror %s
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

#pragma intrinsic(__cpuid)

void test__cpuid_with_array_decayed_to_pointer(int cpuInfo[4], int functionId) {
    __cpuid(cpuInfo, functionId);
}
// CIR-LABEL: __cpuid_with_array_decayed_to_pointer
// CIR: %[[CPUINFO_PTR:.*]] = cir.alloca !cir.ptr<!s32i>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR: {{.*}}cir.cpuid{{.*}}(%{{.*}}, %{{.*}}, %[[ZERO]]) : (!cir.ptr<!s32i>, !s32i, !s32i)

// LLVM-LABEL: __cpuid_with_array_decayed_to_pointer
// LLVM-DAG: [[STACK_PTR_TO_CPU_INFO_PTR:%.*]] = alloca ptr
// LLVM-DAG: [[STACK_PTR_TO_FUNCTION_ID:%.*]] = alloca i32
// LLVM-DAG: [[CPU_INFO_PTR:%.*]] = load ptr, ptr [[STACK_PTR_TO_CPU_INFO_PTR]]
// LLVM-DAG: [[FUNCTION_ID:%.*]] = load i32, ptr [[STACK_PTR_TO_FUNCTION_ID]]
// LLVM-DAG: [[ASMRESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 [[FUNCTION_ID]], i32 0)
// LLVM-DAG: [[RESULT0:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 0
// LLVM-DAG: [[RESULT1:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 1
// LLVM-DAG: [[RESULT2:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 2
// LLVM-DAG: [[RESULT3:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 3
// LLVM-DAG: [[ADDRPTR0:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 0
// LLVM-DAG: [[ADDRPTR1:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 1
// LLVM-DAG: [[ADDRPTR2:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 2
// LLVM-DAG: [[ADDRPTR3:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 3
// LLVM-DAG: store i32 [[RESULT0]], ptr [[ADDRPTR0]], align 4
// LLVM-DAG: store i32 [[RESULT1]], ptr [[ADDRPTR1]], align 4
// LLVM-DAG: store i32 [[RESULT2]], ptr [[ADDRPTR2]], align 4
// LLVM-DAG: store i32 [[RESULT3]], ptr [[ADDRPTR3]], align 4

// OGCG-LABEL: __cpuid_with_array_decayed_to_pointer
// OGCG-DAG: [[STACK_PTR_TO_CPU_INFO_PTR:%.*]] = alloca ptr
// OGCG-DAG: [[STACK_PTR_TO_FUNCTION_ID:%.*]] = alloca i32
// OGCG-DAG: [[ASMRESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 %{{.*}}, i32 0)
// OGCG-DAG: [[RESULT0:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 0
// OGCG-DAG: [[RESULT1:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 1
// OGCG-DAG: [[RESULT2:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 2
// OGCG-DAG: [[RESULT3:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 3
// OGCG-DAG: [[ADDRPTR0:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 0
// OGCG-DAG: [[ADDRPTR1:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 1
// OGCG-DAG: [[ADDRPTR2:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 2
// OGCG-DAG: [[ADDRPTR3:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 3
// OGCG-DAG: store i32 [[RESULT0]], ptr [[ADDRPTR0]], align 4
// OGCG-DAG: store i32 [[RESULT1]], ptr [[ADDRPTR1]], align 4
// OGCG-DAG: store i32 [[RESULT2]], ptr [[ADDRPTR2]], align 4
// OGCG-DAG: store i32 [[RESULT3]], ptr [[ADDRPTR3]], align 4

void test__cpuid_with_array(int functionId) {
    int cpuInfo[4];
    __cpuid(cpuInfo, functionId);
}
// CIR-LABEL: __cpuid_with_array
// CIR: %[[CPUINFO_PTR:.*]] = cir.alloca !cir.array<!s32i x 4>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR: {{.*}}cir.cpuid{{.*}}(%{{.*}}, %{{.*}}, %[[ZERO]]) : (!cir.ptr<!s32i>, !s32i, !s32i)

// LLVM-LABEL: __cpuid_with_array
// LLVM-DAG: [[STACK_PTR_TO_CPU_INFO_PTR:%.*]] = alloca [4 x i32]
// LLVM-DAG: [[STACK_PTR_TO_FUNCTION_ID:%.*]] = alloca i32
// LLVM-DAG: [[CPU_INFO_PTR:%.*]] = getelementptr i32, ptr [[STACK_PTR_TO_CPU_INFO_PTR]]
// LLVM-DAG: [[FUNCTION_ID:%.*]] = load i32, ptr [[STACK_PTR_TO_FUNCTION_ID]]
// LLVM-DAG: [[ASMRESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 [[FUNCTION_ID]], i32 0)
// LLVM-DAG: [[RESULT0:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 0
// LLVM-DAG: [[RESULT1:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 1
// LLVM-DAG: [[RESULT2:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 2
// LLVM-DAG: [[RESULT3:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 3
// LLVM-DAG: [[ADDRPTR0:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 0
// LLVM-DAG: [[ADDRPTR1:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 1
// LLVM-DAG: [[ADDRPTR2:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 2
// LLVM-DAG: [[ADDRPTR3:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 3
// LLVM-DAG: store i32 [[RESULT0]], ptr [[ADDRPTR0]], align 4
// LLVM-DAG: store i32 [[RESULT1]], ptr [[ADDRPTR1]], align 4
// LLVM-DAG: store i32 [[RESULT2]], ptr [[ADDRPTR2]], align 4
// LLVM-DAG: store i32 [[RESULT3]], ptr [[ADDRPTR3]], align 4

// OGCG-LABEL: __cpuid_with_array
// OGCG-DAG: [[STACK_PTR_TO_CPU_INFO_PTR:%.*]] = alloca [4 x i32]
// OGCG-DAG: [[STACK_PTR_TO_FUNCTION_ID:%.*]] = alloca i32
// OGCG-DAG: [[ASMRESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 %{{.*}}, i32 0)
// OGCG-DAG: [[RESULT0:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 0
// OGCG-DAG: [[RESULT1:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 1
// OGCG-DAG: [[RESULT2:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 2
// OGCG-DAG: [[RESULT3:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 3
// OGCG-DAG: [[ADDRPTR0:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 0
// OGCG-DAG: [[ADDRPTR1:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 1
// OGCG-DAG: [[ADDRPTR2:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 2
// OGCG-DAG: [[ADDRPTR3:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 3
// OGCG-DAG: store i32 [[RESULT0]], ptr [[ADDRPTR0]], align 4
// OGCG-DAG: store i32 [[RESULT1]], ptr [[ADDRPTR1]], align 4
// OGCG-DAG: store i32 [[RESULT2]], ptr [[ADDRPTR2]], align 4
// OGCG-DAG: store i32 [[RESULT3]], ptr [[ADDRPTR3]], align 4

#pragma intrinsic(__cpuidex)

void test__cpuidex(int cpuInfo[4], int functionId, int subfunctionId) {
    __cpuidex(cpuInfo, functionId, subfunctionId);
}
// CIR-LABEL: __cpuidex
// CIR: %[[SUBFUNCTION_ID_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["subfunctionId"
// CIR: %[[SUBFUNCTION_ID:.*]] = cir.load {{.*}} %[[SUBFUNCTION_ID_PTR]]
// CIR: {{.*}}cir.cpuid{{.*}}(%{{.*}}, %{{.*}}, %[[SUBFUNCTION_ID]]) : (!cir.ptr<!s32i>, !s32i, !s32i)

// LLVM-LABEL: __cpuidex
// LLVM-DAG: [[STACK_PTR_TO_CPU_INFO_PTR:%.*]] = alloca ptr
// LLVM-DAG: [[STACK_PTR_TO_FUNCTION_ID:%.*]] = alloca i32
// LLVM-DAG: [[STACK_PTR_TO_SUBFUNCTION_ID:%.*]] = alloca i32
// LLVM-DAG: [[CPU_INFO_PTR:%.*]] = load ptr, ptr [[STACK_PTR_TO_CPU_INFO_PTR]]
// LLVM-DAG: [[FUNCTION_ID:%.*]] = load i32, ptr [[STACK_PTR_TO_FUNCTION_ID]]
// LLVM-DAG: [[SUBFUNCTION_ID:%.*]] = load i32, ptr [[STACK_PTR_TO_SUBFUNCTION_ID]]
// LLVM-DAG: [[ASMRESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 [[FUNCTION_ID]], i32 [[SUBFUNCTION_ID]])
// LLVM-DAG: [[RESULT0:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 0
// LLVM-DAG: [[RESULT1:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 1
// LLVM-DAG: [[RESULT2:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 2
// LLVM-DAG: [[RESULT3:%.*]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 3
// LLVM-DAG: [[ADDRPTR0:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 0
// LLVM-DAG: [[ADDRPTR1:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 1
// LLVM-DAG: [[ADDRPTR2:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 2
// LLVM-DAG: [[ADDRPTR3:%.*]] = getelementptr i32, ptr [[CPU_INFO_PTR]], i64 3
// LLVM-DAG: store i32 [[RESULT0]], ptr [[ADDRPTR0]], align 4
// LLVM-DAG: store i32 [[RESULT1]], ptr [[ADDRPTR1]], align 4
// LLVM-DAG: store i32 [[RESULT2]], ptr [[ADDRPTR2]], align 4
// LLVM-DAG: store i32 [[RESULT3]], ptr [[ADDRPTR3]], align 4

// OGCG-LABEL: __cpuidex
// OGCG-DAG: [[STACK_PTR_TO_CPU_INFO_PTR:%.*]] = alloca ptr
// OGCG-DAG: [[STACK_PTR_TO_FUNCTION_ID:%.*]] = alloca i32
// OGCG-DAG: [[STACK_PTR_TO_SUBFUNCTION_ID:%.*]] = alloca i32
// OGCG-DAG: [[ASMRESULTS:%.*]] = call { i32, i32, i32, i32 } asm "{{.*}}cpuid{{.*}}", "={ax},=r,={cx},={dx},0,2"(i32 %{{.*}}, i32 %{{.*}})
// OGCG-DAG: [[RESULT0:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 0
// OGCG-DAG: [[RESULT1:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 1
// OGCG-DAG: [[RESULT2:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 2
// OGCG-DAG: [[RESULT3:%.*]] = extractvalue { i32, i32, i32, i32 } %{{.*}}, 3
// OGCG-DAG: [[ADDRPTR0:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 0
// OGCG-DAG: [[ADDRPTR1:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 1
// OGCG-DAG: [[ADDRPTR2:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 2
// OGCG-DAG: [[ADDRPTR3:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i32 3
// OGCG-DAG: store i32 [[RESULT0]], ptr [[ADDRPTR0]], align 4
// OGCG-DAG: store i32 [[RESULT1]], ptr [[ADDRPTR1]], align 4
// OGCG-DAG: store i32 [[RESULT2]], ptr [[ADDRPTR2]], align 4
// OGCG-DAG: store i32 [[RESULT3]], ptr [[ADDRPTR3]], align 4
