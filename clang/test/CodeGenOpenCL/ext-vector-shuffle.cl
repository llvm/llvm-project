// RUN: %clang_cc1 %s -cl-opt-disable -emit-llvm -o - | not grep 'extractelement'
// RUN: %clang_cc1 %s -cl-opt-disable -emit-llvm -o - | not grep 'insertelement'
// RUN: %clang_cc1 %s -cl-opt-disable -emit-llvm -o - | grep 'shufflevector'

typedef __attribute__(( ext_vector_type(2) )) float float2;
typedef __attribute__(( ext_vector_type(4) )) float float4;

float2 test1(float4 V) {
  return V.xy + V.wz;
}

float4 test2(float4 V) {
  float2 W = V.ww;
  return W.xyxy + W.yxyx;
}

float4 test3(float4 V1, float4 V2) { return (float4)(V1.zw, V2.xy); }
