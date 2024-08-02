// Test that the kernel argument info always refers to SPIR address spaces,
// even if the target has only one address space like x86_64 does.
// RUN: %clang_cc1 -fclangir %s -cl-std=CL2.0 -emit-cir -o - -triple x86_64-unknown-linux-gnu -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR

// RUN: %clang_cc1 -fclangir %s -cl-std=CL2.0 -emit-llvm -o - -triple x86_64-unknown-linux-gnu -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

kernel void foo(__global int * G, __constant int *C, __local int *L) {
  *G = *C + *L;
}
// CIR: cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [1 : i32, 2 : i32, 3 : i32]
// LLVM: !kernel_arg_addr_space ![[MD123:[0-9]+]]
// LLVM: ![[MD123]] = !{i32 1, i32 2, i32 3}
