// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -hlsl-entry main \
// RUN:   -emit-llvm -disable-llvm-passes -o - | \
// RUN: FileCheck %s --check-prefixes=CHECK,DXIL

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-pc-vulkan-compute %s -hlsl-entry main \
// RUN:   -emit-llvm -disable-llvm-passes -o - | \
// RUN: FileCheck %s --check-prefixes=CHECK,SPIRV

// Make sure the entry point is not mangled.
// CHECK:define void @main()
// DXIL:   call void @"?main@@YAXXZ"()
// SPIRV:   call spir_func void @"?main@@YAXXZ"()
// Make sure add function attribute and numthreads attribute.
// CHECK:"hlsl.numthreads"="16,8,1"
// CHECK:"hlsl.shader"="compute"
[numthreads(16,8,1)]
void main() {

}
