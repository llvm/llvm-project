// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// expected-no-diagnostics

typedef int int2 __attribute((ext_vector_type(2)));
typedef int int3 __attribute((ext_vector_type(3)));
typedef int int4 __attribute((ext_vector_type(4)));

__constant int4 i_1_1_1_1 = (int4)(1,2,3,4);
__constant int4 i_2_1_1 = (int4)((int2)(1,2),3,4);
__constant int4 i_1_2_1 = (int4)(1,(int2)(2,3),4);
__constant int4 i_1_1_2 = (int4)(1,2,(int2)(3,4));
__constant int4 i_2_2 = (int4)((int2)(1,2),(int2)(3,4));
__constant int4 i_3_1 = (int4)((int3)(1,2,3),4);
__constant int4 i_1_3 = (int4)(1,(int3)(2,3,4));

typedef float float2 __attribute((ext_vector_type(2)));
typedef float float3 __attribute((ext_vector_type(3)));
typedef float float4 __attribute((ext_vector_type(4)));

__constant float4 f_1_1_1_1 = (float4)(1,2,3,4);
__constant float4 f_2_1_1 = (float4)((float2)(1,2),3,4);
__constant float4 f_1_2_1 = (float4)(1,(float2)(2,3),4);
__constant float4 f_1_1_2 = (float4)(1,2,(float2)(3,4));
__constant float4 f_2_2 = (float4)((float2)(1,2),(float2)(3,4));
__constant float4 f_3_1 = (float4)((float3)(1,2,3),4);
__constant float4 f_1_3 = (float4)(1,(float3)(2,3,4));

