// RUN: %clang_cc1 %s -fclangir -triple spirv64-unknown-unknown -emit-cir -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR

extern __kernel void alias_kernel_function(void)
    __attribute__((alias("kernel_function")));

// CIR-LABEL: cir.func @alias_kernel_function() alias(@kernel_function)

__kernel void kernel_function() {}

// CIR-LABEL: cir.func @kernel_function()
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [], access_qual = [], type = [], base_type = [], type_qual = []>
