// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -O1 -o - %s | FileCheck %s

void foo(__hlsl_resource_t res);

// CHECK: define void @_Z3baru17__hlsl_resource_t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %[[PARAM:[a-zA-Z0-9]+]])
// CHECK: call void @_Z3foou17__hlsl_resource_t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %[[PARAM]])
void bar(__hlsl_resource_t a) {
    foo(a);
}
