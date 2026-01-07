// RUN: %clang_cc1 -triple x86_64-unknown-linux -fms-extensions -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir -Wall -Werror %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

#pragma intrinsic(__cpuid)

void test__cpuid_with_array_decayed_to_pointer(int cpuInfo[4], int function_id) {
    __cpuid(cpuInfo, function_id);
}
// CIR-LABEL: __cpuid_with_array_decayed_to_pointer
// CIR: %[[CPUINFO_PTR:.*]] = cir.alloca !cir.ptr<!s32i>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR: {{.*}}cir.cpuid{{.*}}(%{{.*}}, %{{.*}}, %[[ZERO]]) : (!cir.ptr<!s32i>, !s32i, !s32i)

void test__cpuid_with_array(int function_id) {
    int cpuInfo[4];
    __cpuid(cpuInfo, function_id);
}
// CIR-LABEL: __cpuid_with_array
// CIR: %[[CPUINFO_PTR:.*]] = cir.alloca !cir.array<!s32i x 4>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR: {{.*}}cir.cpuid{{.*}}(%{{.*}}, %{{.*}}, %[[ZERO]]) : (!cir.ptr<!s32i>, !s32i, !s32i)

#pragma intrinsic(__cpuidex)

void test__cpuidex(int cpuInfo[4], int function_id, int subfunction_id) {
    __cpuidex(cpuInfo, function_id, subfunction_id);
}
// CIR-LABEL: __cpuidex
// CIR: %[[SUBFUNCTION_ID_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["subfunction_id"
// CIR: %[[SUBFUNCTION_ID:.*]] = cir.load {{.*}} %[[SUBFUNCTION_ID_PTR]]
// CIR: {{.*}}cir.cpuid{{.*}}(%{{.*}}, %{{.*}}, %[[SUBFUNCTION_ID]]) : (!cir.ptr<!s32i>, !s32i, !s32i)
