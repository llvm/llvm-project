// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -finclude-default-header -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -finclude-default-header -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define void @main()
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[#TOKEN:]] = call token @llvm.experimental.convergence.entry()
// CHECK-NEXT: call{{.*}}void @_Z4mainv() [ "convergencectrl"(token %[[#TOKEN]]) ]

[numthreads(1,1,1)]
void main() {
}

