// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x c -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x c -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c -fclangir -emit-cir %s -o %t.x86.cir
// RUN: FileCheck --input-file=%t.x86.cir %s --check-prefix=X86-CIR --implicit-check-not='cc(spir_kernel)'
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c -fclangir -emit-llvm %s -o %t.x86.ll
// RUN: FileCheck --input-file=%t.x86.ll %s --check-prefix=X86-LLVM --implicit-check-not=spir_kernel

__attribute__((device_kernel)) void k(void) {}

// CIR: cir.func {{.*}} @k() cc(ptx_kernel)
// LLVM: define{{.*}} ptx_kernel void @k()
// X86-CIR: cir.func {{.*}} @k()
// X86-LLVM: define{{.*}} void @k()
