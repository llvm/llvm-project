// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL2.0 -emit-llvm %s -O0 -o - | FileCheck %s

typedef __attribute__(( ext_vector_type(2) ))  int int2;
typedef __attribute__(( ext_vector_type(4) ))  int int4;

// CHECK: %struct.StrucTy = type { i32, i32, i32 }

// CHECK: @GA ={{.*}} addrspace(1) global [6 x [6 x float]] {{[[][[]}}6 x float] [float 1.000000e+00, float 2.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00],
// CHECK:        [6 x float] zeroinitializer, [6 x float] zeroinitializer, [6 x float] zeroinitializer, [6 x float] zeroinitializer, [6 x float] zeroinitializer], align 4 
float GA[6][6]  = {1.0f, 2.0f};

typedef struct {
  int x;
  int y;
  int z;
} StrucTy;

// CHECK: @GS ={{.*}} addrspace(1) global %struct.StrucTy { i32 1, i32 2, i32 0 }, align 4
StrucTy GS = {1, 2};

// CHECK: @GV1 ={{.*}} addrspace(1) global <4 x i32> <i32 1, i32 2, i32 3, i32 4>, align 16
int4 GV1 = (int4)((int2)(1,2),3,4);

// CHECK: @GV2 ={{.*}} addrspace(1) global <4 x i32> <i32 1, i32 1, i32 1, i32 1>, align 16
int4 GV2 = (int4)(1);

// CHECK: @__const.f.S = private unnamed_addr addrspace(2) constant %struct.StrucTy { i32 1, i32 2, i32 0 }, align 4

// CHECK-LABEL: define{{.*}} spir_func void @f()
void f(void) {
  // CHECK: %[[A:.*]] = alloca [6 x [6 x float]], align 4
  // CHECK: %[[S:.*]] = alloca %struct.StrucTy, align 4
  // CHECK: %[[V1:.*]] = alloca <4 x i32>, align 16
  // CHECK: %[[compoundliteral:.*]] = alloca <4 x i32>, align 16
  // CHECK: %[[compoundliteral1:.*]] = alloca <2 x i32>, align 8
  // CHECK: %[[V2:.*]] = alloca <4 x i32>, align 16

  // CHECK: %[[v0:.*]] = bitcast [6 x [6 x float]]* %A to i8*
  // CHECK: call void @llvm.memset.p0i8.i32(i8* align 4 %[[v0]], i8 0, i32 144, i1 false)
  // CHECK: %[[v1:.*]] = bitcast i8* %[[v0]] to [6 x [6 x float]]*
  // CHECK: %[[v2:.*]] = getelementptr inbounds [6 x [6 x float]], [6 x [6 x float]]* %[[v1]], i32 0, i32 0
  // CHECK: %[[v3:.*]] = getelementptr inbounds [6 x float], [6 x float]* %[[v2]], i32 0, i32 0
  // CHECK: store float 1.000000e+00, float* %[[v3]], align 4
  // CHECK: %[[v4:.*]] = getelementptr inbounds [6 x float], [6 x float]* %[[v2]], i32 0, i32 1
  // CHECK: store float 2.000000e+00, float* %[[v4]], align 4
  float A[6][6]  = {1.0f, 2.0f};

  // CHECK: %[[v5:.*]] = bitcast %struct.StrucTy* %S to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %[[v5]], i8 addrspace(2)* align 4 bitcast (%struct.StrucTy addrspace(2)* @__const.f.S to i8 addrspace(2)*), i32 12, i1 false)
  StrucTy S = {1, 2};

  // CHECK: store <2 x i32> <i32 1, i32 2>, <2 x i32>* %[[compoundliteral1]], align 8
  // CHECK: %[[v6:.*]] = load <2 x i32>, <2 x i32>* %[[compoundliteral1]], align 8
  // CHECK: %[[vext:.*]] = shufflevector <2 x i32> %[[v6]], <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  // CHECK: %[[vecinit:.*]] = shufflevector <4 x i32> %[[vext]], <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  // CHECK: %[[vecinit2:.*]] = insertelement <4 x i32> %[[vecinit]], i32 3, i32 2
  // CHECK: %[[vecinit3:.*]] = insertelement <4 x i32> %[[vecinit2]], i32 4, i32 3
  // CHECK: store <4 x i32> %[[vecinit3]], <4 x i32>* %[[compoundliteral]], align 16
  // CHECK: %[[v7:.*]] = load <4 x i32>, <4 x i32>* %[[compoundliteral]], align 16
  // CHECK: store <4 x i32> %[[v7]], <4 x i32>* %[[V1]], align 16
  int4 V1 = (int4)((int2)(1,2),3,4);

  // CHECK: store <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32>* %[[V2]], align 16
  int4 V2 = (int4)(1);
}

