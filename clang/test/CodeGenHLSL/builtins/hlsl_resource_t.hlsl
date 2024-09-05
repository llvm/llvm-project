// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

void foo(__hlsl_resource_t res);

// CHECK: define void @"?bar@@YAXU__hlsl_resource_t@@@Z"(target("dx.TypedBuffer", <4 x float>, 1, 0, 1) %a)
// CHECK: call void @"?foo@@YAXU__hlsl_resource_t@@@Z"(target("dx.TypedBuffer", <4 x float>, 1, 0, 1) %0)
void bar(__hlsl_resource_t a) {
    foo(a);
}
