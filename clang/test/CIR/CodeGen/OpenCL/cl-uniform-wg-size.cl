// RUN: %clang_cc1 -fclangir -triple=spirv64-unknown-unknown -emit-cir -O0 -cl-std=CL1.2 -o %t.cl12.cir %s
// RUN: FileCheck %s -input-file=%t.cl12.cir -check-prefixes CIR,CIR-UNIFORM
// RUN: %clang_cc1 -fclangir -triple=spirv64-unknown-unknown -emit-cir -O0 -cl-std=CL2.0 -o %t.cl20.cir %s
// RUN: FileCheck %s -input-file=%t.cl20.cir -check-prefixes CIR,CIR-NONUNIFORM
// RUN: %clang_cc1 -fclangir -triple=spirv64-unknown-unknown -emit-cir -O0 -cl-std=CL2.0 -cl-uniform-work-group-size -o %t.cl20.uniform1.cir %s
// RUN: FileCheck %s -input-file=%t.cl20.uniform1.cir -check-prefixes CIR,CIR-UNIFORM
// RUN: %clang_cc1 -fclangir -triple=spirv64-unknown-unknown -emit-cir -O0 -cl-std=CL2.0 -foffload-uniform-block -o %t.cl20.uniform2.cir %s
// RUN: FileCheck %s -input-file=%t.cl20.uniform2.cir -check-prefixes CIR,CIR-UNIFORM

// RUN: %clang_cc1 -fclangir -triple=spirv64-unknown-unknown -emit-llvm -O0 -cl-std=CL1.2 -o %t.cl12.ll %s
// RUN: FileCheck %s -input-file=%t.cl12.ll -check-prefixes LLVM,LLVM-UNIFORM
// RUN: %clang_cc1 -fclangir -triple=spirv64-unknown-unknown -emit-llvm -O0 -cl-std=CL2.0 -o %t.cl20.ll %s
// RUN: FileCheck %s -input-file=%t.cl20.ll -check-prefixes LLVM,LLVM-NONUNIFORM
// RUN: %clang_cc1 -fclangir -triple=spirv64-unknown-unknown -emit-llvm -O0 -cl-std=CL2.0 -cl-uniform-work-group-size -o %t.cl20.uniform1.ll %s
// RUN: FileCheck %s -input-file=%t.cl20.uniform1.ll -check-prefixes LLVM,LLVM-UNIFORM
// RUN: %clang_cc1 -fclangir -triple=spirv64-unknown-unknown -emit-llvm -O0 -cl-std=CL2.0 -foffload-uniform-block -o %t.cl20.uniform2.ll %s
// RUN: FileCheck %s -input-file=%t.cl20.uniform2.ll -check-prefixes LLVM,LLVM-UNIFORM

// CIR-LABEL: #fn_attr =
// CIR: cl.kernel = #cir.cl.kernel
// CIR-UNIFORM: cl.uniform_work_group_size = #cir.cl.uniform_work_group_size
// CIR-NONUNIFORM-NOT: cl.uniform_work_group_size = #cir.cl.uniform_work_group_size

// CIR-LABEL: #fn_attr1 =
// CIR-NOT: cl.kernel = #cir.cl.kernel
// CIR-NOT: cl.uniform_work_group_size

kernel void ker() {};
// CIR: cir.func @ker{{.*}} extra(#fn_attr) {
// LLVM: define{{.*}}@ker() #0

void foo() {};
// CIR: cir.func @foo{{.*}} extra(#fn_attr1) {
// LLVM: define{{.*}}@foo() #1

// LLVM-LABEL: attributes #0
// LLVM-UNIFORM: "uniform-work-group-size"="true"
// LLVM-NONUNIFORM: "uniform-work-group-size"="false"

// LLVM-LABEL: attributes #1
// LLVM-NOT: uniform-work-group-size
