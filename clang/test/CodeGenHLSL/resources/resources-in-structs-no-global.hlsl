// When processing a struct containing resources, we should split members recursively into
// their own global variables, and not emit a global of struct or array of
// struct type which contains a resource

// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --implicit-check-not='@g =' --implicit-check-not='@array ='
// RUN: %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --implicit-check-not='@g =' --implicit-check-not='@array ='
// RUN: %if spirv-registered-target && spirv-val %{ %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -O3 -emit-obj -o %t.spv %s && spirv-val --target-env vulkan1.3 %t.spv %}

// CHECK: @_ZL5g.Buf = internal global
// CHECK: @_ZL11array.0.Buf = internal global
// CHECK: @_ZL11array.1.Buf = internal global

struct A {
  RWBuffer<float> Buf;
};
A g;
A array[2];

[numthreads(1, 1, 1)]
void main() {
  g.Buf[0] = 1.0f;
  array[0].Buf[0] = 2.0f;
  array[1].Buf[0] = 3.0f;
}
