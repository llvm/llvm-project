// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -hlsl-entry foo \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Make sure not mangle entry.
// CHECK:define void @foo()
// Make sure add function attribute and numthreads attribute.
// CHECK:"hlsl.numthreads"="16,8,1"
// CHECK:"hlsl.shader"="compute"
[numthreads(16,8,1)]
void foo() {

}
