// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef unsigned int uint4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: define {{.*}}void @clang_shufflevector_v_v(
void clang_shufflevector_v_v( float4* A, float4 x, uint4 mask ) {
// CHECK: [[MASK:%.*]] = and <4 x i32> {{%.*}}, <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[I:%.*]] = extractelement <4 x i32> [[MASK]], i{{[0-9]+}} 0
// CHECK: [[E:%.*]] = extractelement <4 x float> [[X:%.*]], i{{[0-9]+}} [[I]]
//
// Here is where ToT Clang code generation makes a mistake.  
// It uses [[I]] as the insertion index instead of 0.
// Similarly on the remaining insertelement.
// CHECK: [[V:%[a-zA-Z0-9._]+]] = insertelement <4 x float> undef, float [[E]], i{{[0-9]+}} 0

// CHECK: [[I:%.*]] = extractelement <4 x i32> [[MASK]], i{{[0-9]+}} 1
// CHECK: [[E:%.*]] = extractelement <4 x float> [[X]], i{{[0-9]+}} [[I]]
// CHECK: [[V2:%.*]] = insertelement <4 x float> [[V]], float [[E]], i{{[0-9]+}} 1
// CHECK: [[I:%.*]] = extractelement <4 x i32> [[MASK]], i{{[0-9]+}} 2
// CHECK: [[E:%.*]] = extractelement <4 x float> [[X]], i{{[0-9]+}} [[I]]
// CHECK: [[V3:%.*]] = insertelement <4 x float> [[V2]], float [[E]], i{{[0-9]+}} 2
// CHECK: [[I:%.*]] = extractelement <4 x i32> [[MASK]], i{{[0-9]+}} 3
// CHECK: [[E:%.*]] = extractelement <4 x float> [[X]], i{{[0-9]+}} [[I]]
// CHECK: [[V4:%.*]] = insertelement <4 x float> [[V3]], float [[E]], i{{[0-9]+}} 3
// CHECK: store <4 x float> [[V4]], ptr {{%.*}},
  *A = __builtin_shufflevector( x, mask );
}

// CHECK-LABEL: define {{.*}}void @clang_shufflevector_v_v_c(
void clang_shufflevector_v_v_c( float4* A, float4 x, float4 y) {
// CHECK: [[V:%.*]] = shufflevector <4 x float> {{%.*}}, <4 x float> {{%.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK: store <4 x float> [[V]], ptr {{%.*}}
  *A = __builtin_shufflevector( x, y, 0, 4, 1, 5 );
}

// CHECK-LABEL: define {{.*}}void @clang_shufflevector_v_v_undef(
void clang_shufflevector_v_v_undef( float4* A, float4 x, float4 y) {
// CHECK: [[V:%.*]] = shufflevector <4 x float> {{%.*}}, <4 x float> {{%.*}}, <4 x i32> <i32 0, i32 4, i32 undef, i32 5>
// CHECK: store <4 x float> [[V]], ptr {{%.*}}
  *A = __builtin_shufflevector( x, y, 0, 4, -1, 5 );
}
