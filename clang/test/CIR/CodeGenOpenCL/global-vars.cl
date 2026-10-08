// RUN: %clang_cc1 -x cl -triple spir64 -cl-std=CL2.0 -fclangir -emit-cir \
// RUN:   -Wno-deprecated-attributes -mmlir \
// RUN:   --mlir-print-ir-before=cir-target-lowering %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.pre.cir
// RUN: %clang_cc1 -x cl -triple spirv64-unknown-unknown -cl-std=CL2.0 \
// RUN:   -fclangir -emit-llvm -O0 -Wno-deprecated-attributes %s -o %t.cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.cir.ll
// RUN: %clang_cc1 -x cl -triple spirv64-unknown-unknown -cl-std=CL2.0 \
// RUN:   -emit-llvm -O0 -Wno-deprecated-attributes %s -o %t.ogcg.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.ogcg.ll

constant int const_gv = 42;
global int global_gv = 7;
global int tentative_gv;
const sampler_t sampler_gv = 0;

// CIR-DAG: cir.global constant external lang_address_space(offload_constant) @const_gv = #cir.int<42> : !s32i
// CIR-DAG: cir.global external lang_address_space(offload_global) @global_gv = #cir.int<7> : !s32i
// CIR-DAG: cir.global external lang_address_space(offload_global) @tentative_gv = #cir.int<0> : !s32i
// CIR-NOT: @sampler_gv

// LLVM-DAG: @const_gv = addrspace(2) constant i32 42
// LLVM-DAG: @global_gv = addrspace(1) global i32 7
// LLVM-DAG: @tentative_gv = addrspace(1) global i32 0
// LLVM-NOT: @sampler_gv
