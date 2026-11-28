// RUN: %clang_cc1 %s -x cuda -fcuda-is-device -triple spirv64-unknown-unknown -fclangir -emit-cir -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 %s -x hip -fcuda-is-device -triple spirv64-amd-amdhsa -fclangir -emit-cir -o - | FileCheck %s --check-prefix=CIR

extern "C" __attribute__((global, reqd_work_group_size(1, 2, 4),
                          work_group_size_hint(8, 16, 32), vec_type_hint(int)))
void hinted_kernel() {}

// CIR-LABEL: cir.func {{.*}} @hinted_kernel()
// CIR-SAME: cir.cl.kernel_execution_metadata = #cir.cl.kernel_execution_metadata<reqd_work_group_size = [1, 2, 4], work_group_size_hint = [8, 16, 32], vec_type_hint = !s32i

extern "C" __attribute__((device)) void helper() {}

// CIR-LABEL: cir.func {{.*}} @helper()
// CIR-NOT: cir.cl.kernel_execution_metadata
