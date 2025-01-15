// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s

// CHECK-LABEL: ToTwoInts
// CHECK: [[splat:%.*]] = insertelement <1 x i32> poison, i32 {{.*}}, i64 0
// CHECK: [[vec2:%.*]] = shufflevector <1 x i32> [[splat]], <1 x i32> poison, <2 x i32> zeroinitializer
// CHECK: ret <2 x i32> [[vec2]]
int2 ToTwoInts(int V){
  return V.xx;
}

// CHECK-LABEL: ToFourFloats
// [[splat:%.*]] = insertelement <1 x float> poison, float {{.*}}, i64 0
// [[vec4:%.*]] = shufflevector <1 x float> [[splat]], <1 x float> poison, <4 x i32> zeroinitializer
// ret <4 x float> [[vec4]]
float4 ToFourFloats(float V){
  return V.rrrr;
}

// CHECK-LABEL: FillOne
// CHECK: [[vec1Ptr:%.*]] = alloca <1 x i32>, align 4
// CHECK: store <1 x i32> splat (i32 1), ptr [[vec1Ptr]], align 4
// CHECK: [[vec1:%.*]] = load <1 x i32>, ptr [[vec1Ptr]], align 4
// CHECK: [[vec2:%.*]] = shufflevector <1 x i32> [[vec1]], <1 x i32> poison, <2 x i32> zeroinitializer
// CHECK: ret <2 x i32> [[vec2]]
int2 FillOne(){
  return 1.xx;
}

// CHECK-LABEL: FillOneUnsigned
// CHECK: [[vec1Ptr:%.*]] = alloca <1 x i32>, align 4
// CHECK: store <1 x i32> splat (i32 1), ptr [[vec1Ptr]], align 4
// CHECK: [[vec1:%.*]] = load <1 x i32>, ptr [[vec1Ptr]], align 4
// CHECK: [[vec3:%.*]] = shufflevector <1 x i32> [[vec1]], <1 x i32> poison, <3 x i32> zeroinitializer
// CHECK: ret <3 x i32> [[vec3]]
uint3 FillOneUnsigned(){
  return 1u.xxx;
}

// CHECK-LABEL: FillOneUnsignedLong
// CHECK: [[vec1Ptr:%.*]] = alloca <1 x i64>, align 8
// CHECK: store <1 x i64> splat (i64 1), ptr [[vec1Ptr]], align 8
// CHECK: [[vec1:%.*]] = load <1 x i64>, ptr [[vec1Ptr]], align 8
// CHECK: [[vec4:%.*]] = shufflevector <1 x i64> [[vec1]], <1 x i64> poison, <4 x i32> zeroinitializer
// CHECK: ret <4 x i64> [[vec4]]
vector<uint64_t,4> FillOneUnsignedLong(){
  return 1ul.xxxx;
}

// CHECK-LABEL: FillTwoPointFive
// CHECK: [[vec1Ptr:%.*]] = alloca <1 x double>, align 8
// CHECK: store <1 x double> splat (double 2.500000e+00), ptr [[vec1Ptr]], align 8
// CHECK: [[vec1:%.*]] = load <1 x double>, ptr [[vec1Ptr]], align 8
// CHECK: [[vec2:%.*]] = shufflevector <1 x double> [[vec1]], <1 x double> poison, <2 x i32> zeroinitializer
// CHECK: ret <2 x double> [[vec2]]
double2 FillTwoPointFive(){
  return 2.5l.rr;
}

// CHECK-LABEL: FillOneHalf
// CHECK: [[vec1Ptr:%.*]] = alloca <1 x double>, align 8
// CHECK: store <1 x double> splat (double 5.000000e-01), ptr [[vec1Ptr]], align 8
// CHECK: [[vec1:%.*]] = load <1 x double>, ptr [[vec1Ptr]], align 8
// CHECK: [[vec3:%.*]] = shufflevector <1 x double> [[vec1]], <1 x double> poison, <3 x i32> zeroinitializer
// CHECK: ret <3 x double> [[vec3]]
double3 FillOneHalf(){
  return .5l.rrr;
}

// CHECK-LABEL: FillTwoPointFiveFloat
// CHECK: [[vec1Ptr:%.*]] = alloca <1 x float>, align 4
// CHECK: store <1 x float> splat (float 2.500000e+00), ptr [[vec1Ptr]], align 4
// CHECK: [[vec1:%.*]] = load <1 x float>, ptr [[vec1Ptr]], align 4
// CHECK: [[vec4:%.*]] = shufflevector <1 x float> [[vec1]], <1 x float> poison, <4 x i32> zeroinitializer
// CHECK: ret <4 x float> [[vec4]]
float4 FillTwoPointFiveFloat(){
  return 2.5f.rrrr;
}

// The initial codegen for this case is correct but a bit odd. The IR optimizer
// cleans this up very nicely.

// CHECK-LABEL: FillOneHalfFloat
// CHECK: [[vec1Ptr:%.*]] = alloca <1 x float>, align 4
// CHECK: store <1 x float> splat (float 5.000000e-01), ptr [[vec1Ptr]], align 4
// CHECK: [[vec1:%.*]] = load <1 x float>, ptr [[vec1Ptr]], align 4
// CHECK: [[el0:%.*]] = extractelement <1 x float> [[vec1]], i32 0
// CHECK: [[vec1Splat:%.*]] = insertelement <1 x float> poison, float [[el0]], i64 0
// CHECK: [[vec1Ret:%.*]] = shufflevector <1 x float> [[vec1Splat]], <1 x float> poison, <1 x i32> zeroinitializer
// CHECK: ret <1 x float> [[vec1Ret]]
vector<float, 1> FillOneHalfFloat(){
  return .5f.r;
}

// The initial codegen for this case is correct but a bit odd. The IR optimizer
// cleans this up very nicely.

// CHECK-LABEL: HowManyFloats
// CHECK: [[VAddr:%.*]] = alloca float, align 4
// CHECK: [[vec2Ptr:%.*]] = alloca <2 x float>, align 8
// CHECK: [[VVal:%.*]] = load float, ptr [[VAddr]], align 4
// CHECK: [[splat:%.*]] = insertelement <1 x float> poison, float [[VVal]], i64 0
// CHECK: [[vec2:%.*]] = shufflevector <1 x float> [[splat]], <1 x float> poison, <2 x i32> zeroinitializer
// CHECK: store <2 x float> [[vec2]], ptr [[vec2Ptr]], align 8
// CHECK: [[vec2:%.*]] = load <2 x float>, ptr [[vec2Ptr]], align 8
// CHECK: [[vec2Res:%.*]] = shufflevector <2 x float> [[vec2]], <2 x float> poison, <2 x i32> zeroinitializer
// CHECK: ret <2 x float> [[vec2Res]]
float2 HowManyFloats(float V) {
  return V.rr.rr;
}

// This codegen is gnarly because `1.l` is a double, so this creates double
// vectors that need to be truncated down to floats. The optimizer cleans this
// up nicely too.

// CHECK-LABEL: AllRighty
// CHECK: [[Tmp:%.*]] = alloca <1 x double>, align 8
// CHECK: store <1 x double> splat (double 1.000000e+00), ptr [[Tmp]], align 8
// CHECK: [[vec1:%.*]] = load <1 x double>, ptr [[Tmp]], align 8
// CHECK: [[vec3:%.*]] = shufflevector <1 x double> [[vec1]], <1 x double> poison, <3 x i32> zeroinitializer
// CHECK: [[vec3f:%.*]] = fptrunc reassoc nnan ninf nsz arcp afn <3 x double> [[vec3]] to <3 x float>
// CHECK: ret <3 x float> [[vec3f]]

float3 AllRighty() {
  return 1.l.rrr;
}

// CHECK-LABEL: AllRighty2
// CHECK: [[vec1Ptr:%.*]] = alloca <1 x float>, align 4
// CHECK: store <1 x float> splat (float 1.000000e+00), ptr [[vec1Ptr]], align 4
// CHECK: [[vec1:%.*]] = load <1 x float>, ptr [[vec1Ptr]], align 4
// CHECK: [[vec3:%.*]] = shufflevector <1 x float> [[vec1]], <1 x float> poison, <3 x i32>
// CHECK: ret <3 x float> [[vec3]]

float3 AllRighty2() {
  return 1..rrr;
}

// CHECK-LABEL: AssignInt
// CHECK: [[VAddr:%.*]] = alloca i32, align 4
// CHECK: [[XAddr:%.*]] = alloca i32, align 4

// Load V into a vector, then extract V out and store it to X.
// CHECK: [[V:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK: [[Splat:%.*]] = insertelement <1 x i32> poison, i32 [[V]], i64 0
// CHECK: [[VExtVal:%.*]] = extractelement <1 x i32> [[Splat]], i32 0
// CHECK: store i32 [[VExtVal]], ptr [[XAddr]], align 4

// Load V into two separate vectors, then add the extracted X components.
// CHECK: [[V:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK: [[Splat:%.*]] = insertelement <1 x i32> poison, i32 [[V]], i64 0
// CHECK: [[LHS:%.*]] = extractelement <1 x i32> [[Splat]], i32 0

// CHECK: [[V:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK: [[Splat:%.*]] = insertelement <1 x i32> poison, i32 [[V]], i64 0
// CHECK: [[RHS:%.*]] = extractelement <1 x i32> [[Splat]], i32 0

// CHECK: [[Sum:%.*]] = add nsw i32 [[LHS]], [[RHS]]
// CHECK: store i32 [[Sum]], ptr [[XAddr]], align 4
// CHECK: [[X:%.*]] = load i32, ptr [[XAddr]], align 4
// CHECK: ret i32 [[X]]

int AssignInt(int V){
  int X = V.x;
  X.x = V.x + V.x;
  return X;
}
