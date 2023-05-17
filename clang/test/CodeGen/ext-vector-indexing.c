// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

typedef __attribute__(( ext_vector_type(4) )) float float4;
// CHECK: @test
void test(void)
{
  float4 va;
  va.hi[0] = 3.0;
// CHECK:  [[VA:%.*]] = alloca <4 x float>
// CHECK:  [[ADD:%.*]] = getelementptr inbounds float, ptr [[VA]], i64 2
// CHECK:  [[ARRIDX:%.*]] = getelementptr inbounds float, ptr [[ADD]], i64 0
// CHECK:   store float 3.000000e+00, ptr [[ARRIDX]]
}
