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
// CHECK: [[splat:%.*]] = insertelement <1 x float> poison, float {{.*}}, i64 0
// CHECK: [[vec4:%.*]] = shufflevector <1 x float> [[splat]], <1 x float> poison, <4 x i32> zeroinitializer
// ret <4 x float> [[vec4]]
float4 ToFourFloats(float V){
  return V.rrrr;
}

// CHECK-LABEL: ToFourBools
// CHECK: {{%.*}} = zext i1 {{.*}} to i32
// CHECK: [[splat:%.*]] = insertelement <1 x i32> poison, i32 {{.*}}, i64 0
// CHECK-NEXT: [[vec4:%.*]] = shufflevector <1 x i32> [[splat]], <1 x i32> poison, <4 x i32> zeroinitializer
// CHECK-NEXT: [[vec2Ret:%.*]] = trunc <4 x i32> [[vec4]] to <4 x i1>
// CHECK-NEXT: ret <4 x i1> [[vec2Ret]]
bool4 ToFourBools(bool V) {
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

// CHECK-LABEL: FillTrue
// CHECK: [[Tmp:%.*]] = alloca <1 x i32>, align 4
// CHECK-NEXT: store <1 x i32> splat (i32 1), ptr [[Tmp]], align 4
// CHECK-NEXT: [[Vec1:%.*]] = load <1 x i32>, ptr [[Tmp]], align 4
// CHECK-NEXT: [[Vec2:%.*]] = shufflevector <1 x i32> [[Vec1]], <1 x i32> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: [[Vec2Ret:%.*]] = trunc <2 x i32> [[Vec2]] to <2 x i1>
// CHECK-NEXT: ret <2 x i1> [[Vec2Ret]]
bool2 FillTrue() {
  return true.xx;
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

// CHECK-LABEL: HowManyBools
// CHECK: [[VAddr:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Vec2Ptr:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[Tmp:%.*]] = zext i1 {{.*}} to i32
// CHECK-NEXT: store i32 [[Tmp]], ptr [[VAddr]], align 4
// CHECK-NEXT: [[VVal:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NEXT: [[Splat:%.*]] = insertelement <1 x i32> poison, i32 [[VVal]], i64 0
// CHECK-NEXT: [[Vec2:%.*]] = shufflevector <1 x i32> [[Splat]], <1 x i32> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: [[Trunc:%.*]] = trunc <2 x i32> [[Vec2]] to <2 x i1>
// CHECK-NEXT: [[Ext:%.*]] = zext <2 x i1> [[Trunc]] to <2 x i32>
// CHECK-NEXT: store <2 x i32> [[Ext]], ptr [[Vec2Ptr]], align 8
// CHECK-NEXT: [[V2:%.*]] = load <2 x i32>, ptr [[Vec2Ptr]], align 8
// CHECK-NEXT: [[V3:%.*]] = shufflevector <2 x i32> [[V2]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: [[LV1:%.*]] = trunc <2 x i32> [[V3]] to <2 x i1>
// CHECK-NEXT: ret <2 x i1> [[LV1]]
bool2 HowManyBools(bool V) {
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

// CHECK-LABEL: AssignBool
// CHECK: [[VAddr:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[XAddr:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Zext:%.*]] = zext i1 %V to i32
// CHECK-NEXT: store i32 [[Zext]], ptr [[VAddr]], align 4
// CHECK-NEXT: [[X:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NEXT: [[Splat:%.*]] = insertelement <1 x i32> poison, i32 [[X]], i64 0
// CHECK-NEXT: [[Y:%.*]] = extractelement <1 x i32> [[Splat]], i32 0
// CHECK-NEXT: [[Z:%.*]] = trunc i32 [[Y]] to i1
// CHECK-NEXT: [[A:%.*]] = zext i1 [[Z]] to i32
// CHECK-NEXT: store i32 [[A]], ptr [[XAddr]], align 4
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NEXT: [[Splat2:%.*]] = insertelement <1 x i32> poison, i32 [[B]], i64 0
// CHECK-NEXT: [[C:%.*]] = extractelement <1 x i32> [[Splat2]], i32 0
// CHECK-NEXT: [[D:%.*]] = trunc i32 [[C]] to i1
// CHECK-NEXT: br i1 [[D]], label %lor.end, label %lor.rhs

// CHECK: lor.rhs:
// CHECK-NEXT: [[E:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NEXT: [[Splat3:%.*]] = insertelement <1 x i32> poison, i32 [[E]], i64 0
// CHECK-NEXT: [[F:%.*]] = extractelement <1 x i32> [[Splat3]], i32 0
// CHECK-NEXT: [[G:%.*]] = trunc i32 [[F]] to i1
// CHECK-NEXT: br label %lor.end

// CHECK: lor.end:
// CHECK-NEXT: [[H:%.*]] = phi i1 [ true, %entry ], [ [[G]], %lor.rhs ]
// CHECK-NEXT: store i1 [[H]], ptr [[XAddr]], align 4
// CHECK-NEXT: [[I:%.*]] = load i32, ptr [[XAddr]], align 4
// CHECK-NEXT: [[LoadV:%.*]] = trunc i32 [[I]] to i1
// CHECK-NEXT: ret i1 [[LoadV]]
bool AssignBool(bool V) {
  bool X = V.x;
  X.x = V.x || V.x;
  return X;
}

// CHECK-LABEL: AssignBool2
// CHECK: [[VAdddr:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[X:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[Tmp:%.*]] = alloca <1 x i32>, align 4
// CHECK-NEXT: [[SV:%.*]] = zext i1 %V to i32
// CHECK-NEXT: store i32 [[SV]], ptr [[VAddr]], align 4
// CHECK-NEXT: store <1 x i32> splat (i32 1), ptr [[Tmp]], align 4
// CHECK-NEXT: [[Y:%.*]] = load <1 x i32>, ptr [[Tmp]], align 4
// CHECK-NEXT: [[Z:%.*]] = shufflevector <1 x i32> [[Y]], <1 x i32> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: [[LV:%.*]] = trunc <2 x i32> [[Z]] to <2 x i1>
// CHECK-NEXT: [[A:%.*]] = zext <2 x i1> [[LV]] to <2 x i32>
// CHECK-NEXT: store <2 x i32> [[A]], ptr [[X]], align 8
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NEXT: [[LV1:%.*]] = trunc i32 [[B]] to i1
// CHECK-NEXT: [[C:%.*]] = load <2 x i32>, ptr [[X]], align 8
// CHECK-NEXT: [[D:%.*]] = zext i1 [[LV1]] to i32
// CHECK-NEXT: [[E:%.*]] = insertelement <2 x i32> [[C]], i32 [[D]], i32 1
// CHECK-NEXT: store <2 x i32> [[E]], ptr [[X]], align 8
// CHECK-NEXT: ret void
void AssignBool2(bool V) {
  bool2 X = true.xx;
  X.y = V;
}

// CHECK-LABEL: AssignBool3
// CHECK: [[VAddr:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[X:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[Y:%.*]] = zext <2 x i1> %V to <2 x i32>
// CHECK-NEXT: store <2 x i32> [[Y]], ptr [[VAddr]], align 8
// CHECK-NEXT: store <2 x i32> splat (i32 1), ptr [[X]], align 8
// CHECK-NEXT: [[Z:%.*]] = load <2 x i32>, ptr [[VAddr]], align 8
// CHECK-NEXT: [[LV:%.*]] = trunc <2 x i32> [[Z]] to <2 x i1>
// CHECK-NEXT: [[A:%.*]] = load <2 x i32>, ptr [[X]], align 8
// CHECK-NEXT: [[B:%.*]] = zext <2 x i1> [[LV]] to <2 x i32>
// CHECK-NEXT: [[C:%.*]] = shufflevector <2 x i32> [[B]], <2 x i32> poison, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT: store <2 x i32> [[C]], ptr [[X]], align 8
// CHECK-NEXT: ret void
void AssignBool3(bool2 V) {
  bool2 X = {true,true};
  X.xy = V;
}

// CHECK-LABEL: AccessBools
// CHECK: [[X:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT: [[Tmp:%.*]] = alloca <1 x i32>, align 4
// CHECK-NEXT: store <1 x i32> splat (i32 1), ptr [[Tmp]], align 4
// CHECK-NEXT: [[Y:%.*]] = load <1 x i32>, ptr [[Tmp]], align 4
// CHECK-NEXT: [[Z:%.*]] = shufflevector <1 x i32> [[Y]], <1 x i32> poison, <4 x i32> zeroinitializer
// CHECK-NEXT: [[LV:%.*]] = trunc <4 x i32> [[Z]] to <4 x i1>
// CHECK-NEXT: [[A:%.*]] = zext <4 x i1> [[LV]] to <4 x i32>
// CHECK-NEXT: store <4 x i32> [[A]], ptr [[X]], align 16
// CHECK-NEXT: [[B:%.*]] = load <4 x i32>, ptr [[X]], align 16
// CHECK-NEXT: [[C:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> poison, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT: [[LV1:%.*]] = trunc <2 x i32> [[C]] to <2 x i1>
// CHECK-NEXT: ret <2 x i1> [[LV1]]
bool2 AccessBools() {
  bool4 X = true.xxxx;
  return X.zw;
}