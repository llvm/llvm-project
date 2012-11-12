// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef unsigned int uint4 __attribute__((ext_vector_type(4)));

// CHECK: define void @clang_shufflevector_v_v(
void clang_shufflevector_v_v( float4* A, float4 x, uint4 mask ) {
// CHECK: [[MASK:%.*]] = and <4 x i32> {{%.*}}, <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[I:%.*]] = extractelement <4 x i32> [[MASK]], i32 0
// CHECK: [[E:%.*]] = extractelement <4 x float> [[X:%.*]], i32 [[I]]
//
// Here is where ToT Clang code generation makes a mistake.  
// It uses [[I]] as the insertion index instead of 0.
// Similarly on the remaining insertelement.
// CHECK: [[V:%[a-zA-Z0-9._]+]] = insertelement <4 x float> undef, float [[E]], i32 0

// CHECK: [[I:%.*]] = extractelement <4 x i32> [[MASK]], i32 1
// CHECK: [[E:%.*]] = extractelement <4 x float> [[X]], i32 [[I]]
// CHECK: [[V:%.*]] = insertelement <4 x float> [[V]], float [[E]], i32 1
// CHECK: [[I:%.*]] = extractelement <4 x i32> [[MASK]], i32 2
// CHECK: [[E:%.*]] = extractelement <4 x float> [[X]], i32 [[I]]
// CHECK: [[V:%.*]] = insertelement <4 x float> [[V]], float [[E]], i32 2
// CHECK: [[I:%.*]] = extractelement <4 x i32> [[MASK]], i32 3
// CHECK: [[E:%.*]] = extractelement <4 x float> [[X]], i32 [[I]]
// CHECK: [[V:%.*]] = insertelement <4 x float> [[V]], float [[E]], i32 3
// CHECK: store <4 x float> [[V]], <4 x float>* {{%.*}},
  *A = __builtin_shufflevector( x, mask );
}

// CHECK: define void @clang_shufflevector_v_v_c(
void clang_shufflevector_v_v_c( float4* A, float4 x, float4 y, uint4 mask ) {
// CHECK: [[V:%.*]] = shufflevector <4 x float> {{%.*}}, <4 x float> {{%.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK: store <4 x float> [[V]], <4 x float>* {{%.*}}
  *A = __builtin_shufflevector( x, y, 0, 4, 1, 5 );
}
