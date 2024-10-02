// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes -emit-llvm -finclude-default-header -o - %s | FileCheck %s --check-prefixes=CHECK,ALL
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -O3 -emit-llvm -finclude-default-header -o - %s | FileCheck %s --check-prefixes=OPT,ALL

// Case 1: Simple floating integral conversion.
// In this test case a float value is passed to an inout parameter taking an
// integer. It is converted to an integer on call and converted back after the
// function.

// CHECK: define void {{.*}}trunc_Param{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) {{%.*}})
void trunc_Param(inout int X) {}

// ALL-LABEL: define noundef float {{.*}}case1
// CHECK: [[F:%.*]] = alloca float
// CHECK: [[ArgTmp:%.*]] = alloca i32
// CHECK: [[FVal:%.*]] = load float, ptr {{.*}}
// CHECK: [[IVal:%.*]] = fptosi float [[FVal]] to i32
// CHECK: store i32 [[IVal]], ptr [[ArgTmp]]
// CHECK: call void {{.*}}trunc_Param{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) [[ArgTmp]])
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

// CHECK: define void {{.*}}undef{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) {{%.*}})
void undef(out int Z) { }

// ALL-LABEL: define noundef i32 {{.*}}case2
// CHECK: [[V:%.*]] = alloca i32
// CHECK: [[ArgTmp:%.*]] = alloca i32
// CHECK-NOT: store {{.*}}, ptr [[ArgTmp]]
// CHECK: call void {{.*}}unde{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) [[ArgTmp]])
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
// This test should verify that an out parameter value is written to as
// expected.

// CHECK: define void {{.*}}zero{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) {{%.*}})
void zero(out int Z) { Z = 0; }

// ALL-LABEL: define noundef i32 {{.*}}case3
// CHECK: [[V:%.*]] = alloca i32
// CHECK: [[ArgTmp:%.*]] = alloca i32
// CHECK-NOT: store {{.*}}, ptr [[ArgTmp]]
// CHECK: call void {{.*}}zero{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) [[ArgTmp]])
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

// CHECK: define void {{.*}}funky{{.*}}(ptr noalias noundef nonnull align 16 dereferenceable(16) {{%.*}})
void funky(inout int3 X) {
  X.x += 1;
  X.y += 2;
  X.z += 3;
}

// ALL-LABEL: define noundef {{.*}}<3 x i32> {{.*}}case4

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
// CHECK: call void {{.*}}funky{{.*}}(ptr noalias noundef nonnull align 16 dereferenceable(16) [[ArgTmp]])

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

// CHECK: define void {{.*}}increment{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) {{%.*}})
void increment(inout int I) {
  I += 1;
}

// ALL-LABEL: define noundef i32 {{.*}}case5

// CHECK: [[I:%.*]] = alloca i32
// CHECK: [[ArgTmp:%.*]] = alloca i32
// CHECK: store i32 4, ptr [[I]]
// CHECK: [[IInit:%.*]] = load i32, ptr [[I]]
// CHECK: store i32 [[IInit:%.*]], ptr [[ArgTmp]], align 4
// CHECK: call void {{.*}}increment{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) [[ArgTmp]])
// CHECK: [[RetVal:%.*]] = load i32, ptr [[ArgTmp]]
// CHECK: store i32 [[RetVal]], ptr [[I]], align 4
// OPT: ret i32 5
export int case5() {
  int I = 4;
  increment(I);
  return I;
}

// Case 6: Aggregate out parameters.
struct S {
  int X;
  float Y;
};

// CHECK: define void {{.*}}init{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(8) {{%.*}})
void init(out S s) {
  s.X = 3;
  s.Y = 4;
}

// ALL-LABEL: define noundef i32 {{.*}}case6

// CHECK: [[S:%.*]] = alloca %struct.S
// CHECK: [[Tmp:%.*]] = alloca %struct.S
// CHECK: call void {{.*}}init{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(8) [[Tmp]])
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[S]], ptr align 4 [[Tmp]], i32 8, i1 false)

// OPT: ret i32 7
export int case6() {
  S s;
  init(s);
  return s.X + s.Y;
}

// Case 7: Aggregate inout parameters.
struct R {
  int X;
  float Y;
};

// CHECK: define void {{.*}}init{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(8) {{%.*}})
void init(inout R s) {
  s.X = 3;
  s.Y = 4;
}

// ALL-LABEL: define noundef i32 {{.*}}case7

// CHECK: [[S:%.*]] = alloca %struct.R
// CHECK: [[Tmp:%.*]] = alloca %struct.R
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[S]], i32 8, i1 false)
// CHECK: call void {{.*}}init{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(8) [[Tmp]])
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[S]], ptr align 4 [[Tmp]], i32 8, i1 false)

// OPT: ret i32 7
export int case7() {
  R s;
  init(s);
  return s.X + s.Y;
}


// Case 8: Non-scalars with a cast expression.

// CHECK: define void {{.*}}trunc_vec{{.*}}(ptr noalias noundef nonnull align 16 dereferenceable(16) {{%.*}})
void trunc_vec(inout int3 V) {}

// ALL-LABEL: define noundef <3 x float> {{.*}}case8

// CHECK: [[V:%.*]] = alloca <3 x float>
// CHECK: [[Tmp:%.*]] = alloca <3 x i32>
// CHECK: [[FVal:%.*]] = load <3 x float>, ptr [[V]]
// CHECK: [[IVal:%.*]] = fptosi <3 x float> [[FVal]] to <3 x i32>
// CHECK: store <3 x i32> [[IVal]], ptr [[Tmp]]
// CHECK: call void {{.*}}trunc_vec{{.*}}(ptr noalias noundef nonnull align 16 dereferenceable(16) [[Tmp]])
// CHECK: [[IRet:%.*]] = load <3 x i32>, ptr [[Tmp]]
// CHECK: [[FRet:%.*]] = sitofp <3 x i32> [[IRet]] to <3 x float>
// CHECK: store <3 x float> [[FRet]], ptr [[V]]

// OPT: [[IVal:%.*]] = fptosi <3 x float> {{.*}} to <3 x i32>
// OPT: [[FVal:%.*]] = sitofp <3 x i32> [[IVal]] to <3 x float>
// OPT: ret <3 x float> [[FVal]]

export float3 case8(float3 V) {
  trunc_vec(V);
  return V;
}

// Case 9: Side-effecting lvalue argument expression!

void do_nothing(inout int V) {}

// ALL-LABEL: define noundef i32 {{.*}}case9
// CHECK: [[V:%.*]] = alloca i32
// CHECK: [[Tmp:%.*]] = alloca i32
// CHECK: store i32 0, ptr [[V]]
// CHECK: [[VVal:%.*]] = load i32, ptr [[V]]
// CHECK: [[VInc:%.*]] = add nsw i32 [[VVal]], 1
// CHECK: store i32 [[VInc]], ptr [[V]]
// CHECK: [[VArg:%.*]] = load i32, ptr [[V]]
// CHECK-NOT: add
// CHECK: store i32 [[VArg]], ptr [[Tmp]]
// CHECK: call void {{.*}}do_nothing{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) [[Tmp]])
// CHECK: [[RetVal:%.*]] = load i32, ptr [[Tmp]]
// CHECK: store i32 [[RetVal]], ptr [[V]]

// OPT: ret i32 1
export int case9() {
  int V = 0;
  do_nothing(++V);
  return V;
}

// Case 10: Verify argument writeback ordering for aliasing arguments.

void order_matters(inout int X, inout int Y) {
  Y = 2;
  X = 1;
}

// ALL-LABEL: define noundef i32 {{.*}}case10

// CHECK: [[V:%.*]] = alloca i32
// CHECK: [[Tmp0:%.*]] = alloca i32
// CHECK: [[Tmp1:%.*]] = alloca i32
// CHECK: store i32 0, ptr [[V]]
// CHECK: [[VVal:%.*]] = load i32, ptr [[V]]
// CHECK: store i32 [[VVal]], ptr [[Tmp0]]
// CHECK: [[VVal:%.*]] = load i32, ptr [[V]]
// CHECK: store i32 [[VVal]], ptr [[Tmp1]]
// CHECK: call void {{.*}}order_matters{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) [[Tmp1]], ptr noalias noundef nonnull align 4 dereferenceable(4) [[Tmp0]])
// CHECK: [[Arg1Val:%.*]] = load i32, ptr [[Tmp1]]
// CHECK: store i32 [[Arg1Val]], ptr [[V]]
// CHECK: [[Arg2Val:%.*]] = load i32, ptr [[Tmp0]]
// CHECK: store i32 [[Arg2Val]], ptr [[V]]

// OPT: ret i32 2
export int case10() {
  int V = 0;
  order_matters(V, V);
  return V;
}

// Case 11: Verify inout on bitfield lvalues

struct B {
  int X : 8;
  int Y : 8;
};

void setFour(inout int I) {
  I = 4;
}

// ALL-LABEL: define {{.*}} i32 {{.*}}case11

// CHECK: [[B:%.*]] = alloca %struct.B
// CHECK: [[Tmp:%.*]] = alloca i32

// CHECK: [[BFLoad:%.*]] = load i32, ptr [[B]]
// CHECK: [[BFshl:%.*]] = shl i32 [[BFLoad]], 24
// CHECK: [[BFashr:%.*]] = ashr i32 [[BFshl]], 24
// CHECK: store i32 [[BFashr]], ptr [[Tmp]]
// CHECK: call void {{.*}}setFour{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) [[Tmp]])
// CHECK: [[RetVal:%.*]] = load i32, ptr [[Tmp]]
// CHECK: [[BFLoad:%.*]] = load i32, ptr [[B]]
// CHECK: [[BFValue:%.*]] = and i32 [[RetVal]], 255
// CHECK: [[ZerodField:%.*]] = and i32 [[BFLoad]], -256
// CHECK: [[BFSet:%.*]] = or i32 [[ZerodField]], [[BFValue]]
// CHECK: store i32 [[BFSet]], ptr [[B]]

// OPT: ret i32 8
export int case11() {
  B b = {1 , 2};
  setFour(b.X);
  return b.X * b.Y;
}

// Case 12: Uninitialized out parameters are undefined

void oops(out int X) {}
// ALL-LABEL: define {{.*}} i32 {{.*}}case12

// CHECK: [[V:%.*]] = alloca i32
// CHECK: [[Tmp:%.*]] = alloca i32
// CHECK-NOT: store {{.*}}, ptr [[Tmp]]
// CHECK: call void {{.*}}oops{{.*}}(ptr noalias noundef nonnull align 4 dereferenceable(4) [[Tmp]])
// CHECK: [[ArgVal:%.*]] = load i32, ptr [[Tmp]]
// CHECK: store i32 [[ArgVal]], ptr [[V]]

// OPT:  ret i32 undef
export int case12() {
  int V = 0;
  oops(V);
  return V;
}
