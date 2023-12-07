// RUN: %clang_cc1 -emit-llvm %s -o - -O0 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -cl-std=clc++ -O0 | FileCheck %s

typedef __attribute__((ext_vector_type(2))) int int2;
typedef __attribute__((ext_vector_type(3))) int int3;
typedef __attribute__((ext_vector_type(4)))  int int4;
typedef __attribute__((ext_vector_type(8)))  int int8;
typedef __attribute__((ext_vector_type(4))) float float4;

__constant const int4 c1 = (int4)(1, 2, ((int2)(3)));
// CHECK: constant <4 x i32> <i32 1, i32 2, i32 3, i32 3>

__constant const int4 c2 = (int4)(1, 2, ((int2)(3, 4)));
// CHECK: constant <4 x i32> <i32 1, i32 2, i32 3, i32 4>

void vector_literals_valid() {
  //CHECK: insertelement <4 x i32> <i32 1, i32 2, i32 undef, i32 undef>, i32 %{{.+}}, i32 2
  //CHECK: insertelement <4 x i32> %{{.+}}, i32 %{{.+}}, i32 3
  int4 a_1_1_1_1 = (int4)(1, 2, c1.s2, c2.s3);

  //CHECK: store <2 x i32> <i32 1, i32 2>, ptr
  //CHECK: shufflevector <2 x i32> %{{[0-9]+}}, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  //CHECK: shufflevector <4 x i32> %{{.+}}, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  //CHECK: insertelement <4 x i32> %{{.+}}, i32 3, i32 2
  //CHECK: insertelement <4 x i32> %{{.+}}, i32 4, i32 3
  int4 a_2_1_1 = (int4)((int2)(1, 2), 3, 4);

  //CHECK: store <2 x i32> <i32 2, i32 3>, ptr
  //CHECK: shufflevector <2 x i32> %{{[0-9]+}}, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  //CHECK: shufflevector <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>, <4 x i32> %{{.+}}, <4 x i32> <i32 0, i32 4, i32 5, i32 poison>
  //CHECK: insertelement <4 x i32> %{{.+}}, i32 4, i32 3
  int4 a_1_2_1 = (int4)(1, (int2)(2, 3), 4);

  //CHECK: store <2 x i32> <i32 3, i32 4>, ptr
  //CHECK: shufflevector <2 x i32> %{{[0-9]+}}, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  //CHECK: shufflevector <4 x i32> <i32 1, i32 2, i32 undef, i32 undef>, <4 x i32> %{{.+}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  int4 a_1_1_2 = (int4)(1, 2, (int2)(3, 4));

  //CHECK: store <2 x i32> <i32 1, i32 2>, ptr
  //CHECK: shufflevector <2 x i32> %{{[0-9]+}}, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  //CHECK: shufflevector <4 x i32> %{{.+}}, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  //CHECK: shufflevector <4 x i32> %{{.+}}, <4 x i32> <i32 3, i32 3, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  int4 a_2_2 = (int4)((int2)(1, 2), (int2)(3));

  //CHECK: store <4 x i32> <i32 2, i32 3, i32 4, i32 undef>, ptr
  //CHECK: shufflevector <4 x i32> %{{.+}}, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
  //CHECK: shufflevector <3 x i32> %{{.+}}, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  //CHECK: shufflevector <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>, <4 x i32> %{{.+}}, <4 x i32> <i32 0, i32 4, i32 5, i32 6>
  int4 a_1_3 = (int4)(1, (int3)(2, 3, 4));

  //CHECK: store <4 x i32> <i32 1, i32 1, i32 1, i32 1>, ptr %a
  int4 a = (int4)(1);

  //CHECK: load <4 x i32>, ptr %a
  //CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  //CHECK: shufflevector <2 x i32> %{{[0-9]+}}, <2 x i32> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  //CHECK: shufflevector <8 x i32> <i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>, <8 x i32> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison>
  //CHECK: load <4 x i32>, ptr %a
  //CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  //CHECK: shufflevector <8 x i32> %{{.+}}, <8 x i32> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  int8 b = (int8)(1, 2, a.xy, a);

  //CHECK: store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, ptr %V2
  float4 V2 = (float4)(1);
}

void vector_literals_with_cast() {
  // CHECK-LABEL: vector_literals_with_cast
  // CHECK: store <2 x i32> <i32 12, i32 34>, ptr
  // CHECK: extractelement <2 x i32> %{{[0-9]+}}, i{{[0-9]+}} 0
  unsigned int withCast = ((int2)((int2)(12, 34))).s0;
}
