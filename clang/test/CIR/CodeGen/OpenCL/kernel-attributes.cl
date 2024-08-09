// RUN: %clang_cc1 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

typedef unsigned int uint4 __attribute__((ext_vector_type(4)));


kernel  __attribute__((vec_type_hint(int))) __attribute__((reqd_work_group_size(1,2,4))) void kernel1(int a) {}

// CIR-DAG: #fn_attr[[KERNEL1:[0-9]*]] = {{.+}}cl.kernel_metadata = #cir.cl.kernel_metadata<reqd_work_group_size = [1 : i32, 2 : i32, 4 : i32], vec_type_hint = !s32i, vec_type_hint_signedness = 1>{{.+}}
// CIR-DAG: cir.func @kernel1{{.+}} extra(#fn_attr[[KERNEL1]])

// LLVM-DAG: define {{(dso_local )?}}spir_kernel void @kernel1(i32 {{[^%]*}}%0) {{[^{]+}} !reqd_work_group_size ![[MD1_REQD_WG:[0-9]+]] !vec_type_hint ![[MD1_VEC_TYPE:[0-9]+]]
// LLVM-DAG: [[MD1_VEC_TYPE]] = !{i32 undef, i32 1}
// LLVM-DAG: [[MD1_REQD_WG]] = !{i32 1, i32 2, i32 4}


kernel __attribute__((vec_type_hint(uint4))) __attribute__((work_group_size_hint(8,16,32))) void kernel2(int a) {}

// CIR-DAG: #fn_attr[[KERNEL2:[0-9]*]] = {{.+}}cl.kernel_metadata = #cir.cl.kernel_metadata<work_group_size_hint = [8 : i32, 16 : i32, 32 : i32], vec_type_hint = !cir.vector<!u32i x 4>, vec_type_hint_signedness = 0>{{.+}}
// CIR-DAG: cir.func @kernel2{{.+}} extra(#fn_attr[[KERNEL2]])

// LLVM-DAG: define {{(dso_local )?}}spir_kernel void @kernel2(i32 {{[^%]*}}%0) {{[^{]+}} !vec_type_hint ![[MD2_VEC_TYPE:[0-9]+]] !work_group_size_hint ![[MD2_WG_SIZE:[0-9]+]]
// LLVM-DAG: [[MD2_VEC_TYPE]] = !{<4 x i32> undef, i32 0}
// LLVM-DAG: [[MD2_WG_SIZE]] = !{i32 8, i32 16, i32 32}


kernel __attribute__((intel_reqd_sub_group_size(8))) void kernel3(int a) {}

// CIR-DAG: #fn_attr[[KERNEL3:[0-9]*]] = {{.+}}cl.kernel_metadata = #cir.cl.kernel_metadata<intel_reqd_sub_group_size = 8 : i32>{{.+}}
// CIR-DAG: cir.func @kernel3{{.+}} extra(#fn_attr[[KERNEL3]])

// LLVM-DAG: define {{(dso_local )?}}spir_kernel void @kernel3(i32 {{[^%]*}}%0) {{[^{]+}} !intel_reqd_sub_group_size ![[MD3_INTEL:[0-9]+]]
// LLVM-DAG: [[MD3_INTEL]] = !{i32 8}
