// RUN: %clang_cc1 %s -x cuda -fcuda-is-device -triple spirv64-unknown-unknown -fclangir -emit-cir -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 %s -x hip -fcuda-is-device -triple spirv64-amd-amdhsa -fclangir -emit-cir -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 %s -x cuda -fcuda-is-device -triple spirv64-unknown-unknown -fclangir -emit-llvm -o - | FileCheck %s --check-prefixes=LLVM,CIR-LLVM
// RUN: %clang_cc1 %s -x hip -fcuda-is-device -triple spirv64-amd-amdhsa -fclangir -emit-llvm -o - | FileCheck %s --check-prefixes=LLVM,CIR-LLVM
// RUN: %clang_cc1 %s -x cuda -fcuda-is-device -triple spirv64-unknown-unknown -emit-llvm -o - | FileCheck %s --check-prefixes=LLVM,OGCG
// RUN: %clang_cc1 %s -x hip -fcuda-is-device -triple spirv64-amd-amdhsa -emit-llvm -o - | FileCheck %s --check-prefixes=LLVM,OGCG

extern "C" __attribute__((global, reqd_work_group_size(1, 2, 4),
                          work_group_size_hint(8, 16, 32), vec_type_hint(int)))
void hinted_kernel() {}

// CIR-LABEL: cir.func {{.*}} @hinted_kernel()
// CIR-SAME: cir.cl.kernel_execution_metadata = #cir.cl.kernel_execution_metadata<reqd_work_group_size = [1, 2, 4], work_group_size_hint = [8, 16, 32], vec_type_hint = !s32i
// LLVM-LABEL: define spir_kernel void @hinted_kernel()
// LLVM-SAME: !vec_type_hint ![[VEC:[0-9]+]]
// LLVM-SAME: !work_group_size_hint ![[HINT:[0-9]+]]
// LLVM-SAME: !reqd_work_group_size ![[REQD:[0-9]+]]

extern "C" __attribute__((device)) void helper() {}

// CIR-LABEL: cir.func {{.*}} @helper()
// CIR-NOT: cir.cl.kernel_execution_metadata
// LLVM-LABEL: define spir_func void @helper()
// LLVM-NOT: !vec_type_hint
// LLVM-NOT: !work_group_size_hint
// LLVM-NOT: !reqd_work_group_size

// CIR-LLVM-DAG: ![[VEC]] = !{i32 undef, i32 1}
// OGCG-DAG: ![[VEC]] = !{i32 poison, i32 1}
// LLVM-DAG: ![[HINT]] = !{i32 8, i32 16, i32 32}
// LLVM-DAG: ![[REQD]] = !{i32 1, i32 2, i32 4}
