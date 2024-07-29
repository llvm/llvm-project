// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes -emit-llvm -finclude-default-header -o - %s | FileCheck %s --check-prefixes=CHECK,ALL
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -O3 -emit-llvm -finclude-default-header -o - %s | FileCheck %s --check-prefixes=OPT,ALL

// Case 1: Simple floating integral conversion.
// In this test case a float value is passed to an inout parameter taking an
// integer. It is converted to an integer on call and converted back after the
// function.
void trunc_Param(inout int X) {}

// ALL-LABEL: define noundef float {{.*}}case1
// CHECK: [[F:%.*]] = alloca float
// CHECK: [[ArgTmp:%.*]] = alloca i32
// CHECK: [[FVal:%.*]] = load float, ptr {{.*}}
// CHECK: [[IVal:%.*]] = fptosi float [[FVal]] to i32
// CHECK: store i32 [[IVal]], ptr [[ArgTmp]]
// CHECK: call void {{.*}}trunc_Param{{.*}}(ptr noundef nonnull align 4 dereferenceable(4) [[ArgTmp]])
// CHECK: [[IRet:%.*]] = load i32, ptr [[ArgTmp]]
// CHECK: [[FRet:%.*]] = sitofp i32 [[IRet]] to float
// CHECK: store float [[FRet]], ptr [[F]]
// OPT: [[IVal:%.*]] = fptosi float {{.*}} to i32
// OPT: [[FVal:%.*]] = sitofp i32 [[IVal]] to float
// OPT: ret float [[FVal]]
export float case1(float F) {
  trunc_Param(F);
  return F;
}

// Case 2: Uninitialized `out` parameters.
// `out` parameters are not pre-initialized by the caller, so they are
// uninitialized in the function. If they are not initialized before the
// function returns the value is undefined.
void undef(out int Z) { }

// ALL-LABEL: define noundef i32 {{.*}}case2
// CHECK: [[V:%.*]] = alloca i32
// CHECK: [[ArgTmp:%.*]] = alloca i32
// CHECK-NOT: store {{.*}}, ptr [[ArgTmp]]
// CHECK: call void {{.*}}unde{{.*}}(ptr noundef nonnull align 4 dereferenceable(4) [[ArgTmp]])
// CHECK-NOT: store {{.*}}, ptr [[ArgTmp]]
// CHECK: [[Res:%.*]] = load i32, ptr [[ArgTmp]]
// CHECK: store i32 [[Res]], ptr [[V]], align 4
// OPT: ret i32 undef
export int case2() {
  int V;
  undef(V);
  return V;
}

// Case 3: Simple initialized `out` parameter.
// This test should verify that an out parameter value is written to as expected.
void zero(out int Z) { Z = 0; }

// ALL-LABEL: define noundef i32 {{.*}}case3
// CHECK: [[V:%.*]] = alloca i32
// CHECK: [[ArgTmp:%.*]] = alloca i32
// CHECK-NOT: store {{.*}}, ptr [[ArgTmp]]
// CHECK: call void {{.*}}zero{{.*}}(ptr noundef nonnull align 4 dereferenceable(4) [[ArgTmp]])
// CHECK-NOT: store {{.*}}, ptr [[ArgTmp]]
// CHECK: [[Res:%.*]] = load i32, ptr [[ArgTmp]]
// CHECK: store i32 [[Res]], ptr [[V]], align 4
// OPT: ret i32 0
export int case3() {
  int V;
  zero(V);
  return V;
}

// Case 4: Vector swizzle arguments.
// Vector swizzles in HLSL produce lvalues, so they can be used as arguments to
// inout parameters and the swizzle is reversed on writeback.
void funky(inout int3 X) {
  X.x += 1;
  X.y += 2;
  X.z += 3;
}

// ALL-LABEL: define noundef <3 x i32> {{.*}}case4

// This block initializes V = 0.xxx.
// CHECK:  [[V:%.*]] = alloca <3 x i32>
// CHECK:  [[ArgTmp:%.*]] = alloca <3 x i32>
// CHECK:  store <1 x i32> zeroinitializer, ptr [[ZeroPtr:%.*]]
// CHECK:  [[ZeroV1:%.*]] = load <1 x i32>, ptr [[ZeroPtr]]
// CHECK:  [[ZeroV3:%.*]] = shufflevector <1 x i32> [[ZeroV1]], <1 x i32> poison, <3 x i32> zeroinitializer
// CHECK:  store <3 x i32> [[ZeroV3]], ptr [[V]]

// Shuffle the vector to the temporary.
// CHECK:  [[VVal:%.*]] = load <3 x i32>, ptr [[V]]
// CHECK:  [[Vyzx:%.*]] = shufflevector <3 x i32> [[VVal]], <3 x i32> poison, <3 x i32> <i32 1, i32 2, i32 0>
// CHECK:  store <3 x i32> [[Vyzx]], ptr [[ArgTmp]]

// Call the function with the temporary.
// CHECK: call void {{.*}}funky{{.*}}(ptr noundef nonnull align 16 dereferenceable(16) [[ArgTmp]])

// Shuffle it back.
// CHECK:  [[RetVal:%.*]] = load <3 x i32>, ptr [[ArgTmp]]
// CHECK:  [[Vxyz:%.*]] = shufflevector <3 x i32> [[RetVal]], <3 x i32> poison, <3 x i32> <i32 2, i32 0, i32 1>
// CHECK:  store <3 x i32> [[Vxyz]], ptr [[V]]

// OPT: ret <3 x i32> <i32 3, i32 1, i32 2>
export int3 case4() {
  int3 V = 0.xxx;
  funky(V.yzx);
  return V;
}


// Case 5: Straightforward inout of a scalar value.
void increment(inout int I) {
  I += 1;
}

// ALL-LABEL: define noundef i32 {{.*}}case5

// CHECK: [[I:%.*]] = alloca i32
// CHECK: [[ArgTmp:%.*]] = alloca i32
// CHECK: store i32 4, ptr [[I]]
// CHECK: [[IInit:%.*]] = load i32, ptr [[I]]
// CHECK: store i32 [[IInit:%.*]], ptr [[ArgTmp]], align 4
// CHECK: call void {{.*}}increment{{.*}}(ptr noundef nonnull align 4 dereferenceable(4) [[ArgTmp]])
// CHECK: [[RetVal:%.*]] = load i32, ptr [[ArgTmp]]
// CHECK: store i32 [[RetVal]], ptr [[I]], align 4
// OPT: ret i32 5
export int case5() {
  int I = 4;
  increment(I);
  return I;
}
