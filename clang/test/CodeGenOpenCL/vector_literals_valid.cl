// RUN: %clang_cc1 -emit-llvm %s -o %t

typedef __attribute__(( ext_vector_type(2) ))  int int2;
typedef __attribute__(( ext_vector_type(3) ))  int int3;
typedef __attribute__(( ext_vector_type(4) ))  int int4;
typedef __attribute__(( ext_vector_type(8) ))  int int8;
typedef __attribute__(( ext_vector_type(4) ))  float float4;

void vector_literals_valid() {
  int4 a_1_1_1_1 = (int4)(1,2,3,4);
  int4 a_2_1_1 = (int4)((int2)(1,2),3,4);
  int4 a_1_2_1 = (int4)(1,(int2)(2,3),4);
  int4 a_1_1_2 = (int4)(1,2,(int2)(3,4));
  int4 a_2_2 = (int4)((int2)(1,2),(int2)(3,4));
  int4 a_3_1 = (int4)((int3)(1,2,3),4);
  int4 a_1_3 = (int4)(1,(int3)(2,3,4));
  int4 a = (int4)(1);
  int8 b = (int8)(1,2,a.xy,a);
  float4 V2 = (float4) (1);
}


