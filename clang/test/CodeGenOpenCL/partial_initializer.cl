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

// CHECK: @GV2 ={{.*}} addrspace(1) global <4 x i32> splat (i32 1), align 16
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

  // CHECK: call void @llvm.memset.p0.i32(ptr align 4 %A, i8 0, i32 144, i1 false)
  // CHECK: %[[v2:.*]] = getelementptr inbounds [6 x [6 x float]], ptr %A, i32 0, i32 0
  // CHECK: %[[v3:.*]] = getelementptr inbounds [6 x float], ptr %[[v2]], i32 0, i32 0
  // CHECK: store float 1.000000e+00, ptr %[[v3]], align 4
  // CHECK: %[[v4:.*]] = getelementptr inbounds [6 x float], ptr %[[v2]], i32 0, i32 1
  // CHECK: store float 2.000000e+00, ptr %[[v4]], align 4
  float A[6][6]  = {1.0f, 2.0f};

  // CHECK: call void @llvm.memcpy.p0.p2.i32(ptr align 4 %S, ptr addrspace(2) align 4 @__const.f.S, i32 12, i1 false)
  StrucTy S = {1, 2};

  // CHECK: store <2 x i32> <i32 1, i32 2>, ptr %[[compoundliteral1]], align 8
  // CHECK: %[[v6:.*]] = load <2 x i32>, ptr %[[compoundliteral1]], align 8
  // CHECK: %[[vext:.*]] = shufflevector <2 x i32> %[[v6]], <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  // CHECK: %[[vecinit:.*]] = shufflevector <4 x i32> %[[vext]], <4 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  // CHECK: %[[vecinit2:.*]] = insertelement <4 x i32> %[[vecinit]], i32 3, i32 2
  // CHECK: %[[vecinit3:.*]] = insertelement <4 x i32> %[[vecinit2]], i32 4, i32 3
  // CHECK: store <4 x i32> %[[vecinit3]], ptr %[[compoundliteral]], align 16
  // CHECK: %[[v7:.*]] = load <4 x i32>, ptr %[[compoundliteral]], align 16
  // CHECK: store <4 x i32> %[[v7]], ptr %[[V1]], align 16
  int4 V1 = (int4)((int2)(1,2),3,4);

  // CHECK: store <4 x i32> splat (i32 1), ptr %[[V2]], align 16
  int4 V2 = (int4)(1);
}

