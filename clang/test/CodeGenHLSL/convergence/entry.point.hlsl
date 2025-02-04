// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -finclude-default-header -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define void @main()
// CHECK-NEXT: entry:
// CHECK-NEXT: [[token:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK-NEXT: call spir_func void @_Z4mainv() [ "convergencectrl"(token [[token]]) ]

[numthreads(1,1,1)]
void main() {
}

