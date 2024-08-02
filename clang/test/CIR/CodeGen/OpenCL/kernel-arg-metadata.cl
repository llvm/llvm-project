// RUN: %clang_cc1 %s -fclangir -triple spirv64-unknown-unknown -emit-cir -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 %s -fclangir -triple spirv64-unknown-unknown -emit-llvm -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

__kernel void kernel_function() {}

// CIR: #fn_attr[[ATTR:[0-9]*]] = {{.+}}cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [], access_qual = [], type = [], base_type = [], type_qual = []>{{.+}}
// CIR: cir.func @kernel_function() extra(#fn_attr[[ATTR]])

// LLVM: define {{.*}}void @kernel_function() {{[^{]+}} !kernel_arg_addr_space ![[MD:[0-9]+]] !kernel_arg_access_qual ![[MD]] !kernel_arg_type ![[MD]] !kernel_arg_base_type ![[MD]] !kernel_arg_type_qual ![[MD]] {
// LLVM: ![[MD]] = !{}
