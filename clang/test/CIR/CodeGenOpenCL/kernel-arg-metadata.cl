// RUN: %clang_cc1 %s -fclangir -triple spirv64-unknown-unknown -emit-cir -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 %s -fclangir -triple spirv64-unknown-unknown -emit-llvm -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 %s -triple spirv64-unknown-unknown -emit-llvm -o %t.ogcg.ll
// RUN: FileCheck %s --input-file=%t.ogcg.ll --check-prefix=LLVM

extern __kernel void alias_kernel_function(void)
    __attribute__((alias("kernel_function")));

// CIR-LABEL: cir.func @alias_kernel_function() alias(@kernel_function)

__kernel void kernel_function() {}

// CIR-LABEL: cir.func @kernel_function()
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [], access_qual = [], type = [], base_type = [], type_qual = []>

// LLVM-LABEL: define spir_kernel void @kernel_function()
// LLVM-SAME: !kernel_arg_addr_space ![[EMPTY_ARG_METADATA:[0-9]+]]
// LLVM-SAME: !kernel_arg_access_qual ![[EMPTY_ARG_METADATA]]
// LLVM-SAME: !kernel_arg_type ![[EMPTY_ARG_METADATA]]
// LLVM-SAME: !kernel_arg_base_type ![[EMPTY_ARG_METADATA]]
// LLVM-SAME: !kernel_arg_type_qual ![[EMPTY_ARG_METADATA]]
// LLVM: ![[EMPTY_ARG_METADATA]] = !{}
