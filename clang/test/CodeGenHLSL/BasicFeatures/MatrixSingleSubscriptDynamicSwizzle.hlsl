// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.7-library -disable-llvm-passes -emit-llvm -finclude-default-header -o - %s | FileCheck %s
// XFAIL: *

void setMatrix(out float4x4 M, int index, float4 V) {
    M[index].abgr = V;
}

float3 getMatrix(float4x4 M, int index) {
    return M[index].rgb;
}
