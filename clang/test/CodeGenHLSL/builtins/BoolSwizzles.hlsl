// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -load-bool-from-mem=truncate -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck -check-prefixes=CHECK,CHECK-TR %s
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -load-bool-from-mem=nonzero -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck -check-prefixes=CHECK,CHECK-NZ %s

// CHECK-LABEL: ToFourBools
// CHECK: {{%.*}} = zext i1 {{.*}} to i32
// CHECK: [[splat:%.*]] = insertelement <1 x i32> poison, i32 {{.*}}, i64 0
// CHECK-NEXT: [[vec4:%.*]] = shufflevector <1 x i32> [[splat]], <1 x i32> poison, <4 x i32> zeroinitializer
// CHECK-NZ-NEXT: [[vec2Ret:%.*]] = icmp ne <4 x i32> [[vec4]], zeroinitializer
// CHECK-TR-NEXT: [[vec2Ret:%.*]] = trunc <4 x i32> [[vec4]] to <4 x i1>
// CHECK-NEXT: ret <4 x i1> [[vec2Ret]]
bool4 ToFourBools(bool V) {
  return V.rrrr;
}

// CHECK-LABEL: FillTrue
// CHECK: [[Tmp:%.*]] = alloca <1 x i32>, align 4
// CHECK-NEXT: store <1 x i32> splat (i32 1), ptr [[Tmp]], align 4
// CHECK-NEXT: [[Vec1:%.*]] = load <1 x i32>, ptr [[Tmp]], align 4
// CHECK-NEXT: [[Vec2:%.*]] = shufflevector <1 x i32> [[Vec1]], <1 x i32> poison, <2 x i32> zeroinitializer
// CHECK-NZ-NEXT: [[Vec2Ret:%.*]] = icmp ne <2 x i32> [[Vec2]], zeroinitializer
// CHECK-TR-NEXT: [[Vec2Ret:%.*]] = trunc <2 x i32> [[Vec2]] to <2 x i1>
// CHECK-NEXT: ret <2 x i1> [[Vec2Ret]]
bool2 FillTrue() {
  return true.xx;
}

// CHECK-LABEL: HowManyBools
// CHECK: [[VAddr:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Vec2Ptr:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[Tmp:%.*]] = zext i1 {{.*}} to i32
// CHECK-NEXT: store i32 [[Tmp]], ptr [[VAddr]], align 4
// CHECK-NEXT: [[VVal:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NEXT: [[Splat:%.*]] = insertelement <1 x i32> poison, i32 [[VVal]], i64 0
// CHECK-NEXT: [[Vec2:%.*]] = shufflevector <1 x i32> [[Splat]], <1 x i32> poison, <2 x i32> zeroinitializer
// CHECK-NZ-NEXT: [[Trunc:%.*]] = icmp ne <2 x i32> [[Vec2]], zeroinitializer
// CHECK-TR-NEXT: [[Trunc:%.*]] = trunc <2 x i32> [[Vec2]] to <2 x i1>
// CHECK-NEXT: [[Ext:%.*]] = zext <2 x i1> [[Trunc]] to <2 x i32>
// CHECK-NEXT: store <2 x i32> [[Ext]], ptr [[Vec2Ptr]], align 8
// CHECK-NEXT: [[V2:%.*]] = load <2 x i32>, ptr [[Vec2Ptr]], align 8
// CHECK-NEXT: [[V3:%.*]] = shufflevector <2 x i32> [[V2]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK-NZ-NEXT: [[LV1:%.*]] = icmp ne <2 x i32> [[V3]], zeroinitializer
// CHECK-TR-NEXT: [[LV1:%.*]] = trunc <2 x i32> [[V3]] to <2 x i1>
// CHECK-NEXT: ret <2 x i1> [[LV1]]
bool2 HowManyBools(bool V) {
  return V.rr.rr;
}

// CHECK-LABEL: AssignBool
// CHECK: [[VAddr:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[XAddr:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Zext:%.*]] = zext i1 %V to i32
// CHECK-NEXT: store i32 [[Zext]], ptr [[VAddr]], align 4
// CHECK-NEXT: [[X:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NEXT: [[Splat:%.*]] = insertelement <1 x i32> poison, i32 [[X]], i64 0
// CHECK-NEXT: [[Y:%.*]] = extractelement <1 x i32> [[Splat]], i32 0
// CHECK-NZ-NEXT: [[Z:%.*]] = icmp ne i32 [[Y]], 0
// CHECK-TR-NEXT: [[Z:%.*]] = trunc i32 [[Y]] to i1
// CHECK-NEXT: [[A:%.*]] = zext i1 [[Z]] to i32
// CHECK-NEXT: store i32 [[A]], ptr [[XAddr]], align 4
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NEXT: [[Splat2:%.*]] = insertelement <1 x i32> poison, i32 [[B]], i64 0
// CHECK-NEXT: [[C:%.*]] = extractelement <1 x i32> [[Splat2]], i32 0
// CHECK-NZ-NEXT: [[D:%.*]] = icmp ne i32 [[C]], 0
// CHECK-TR-NEXT: [[D:%.*]] = trunc i32 [[C]] to i1
// CHECK-NEXT: br i1 [[D]], label %lor.end, label %lor.rhs

// CHECK: lor.rhs:
// CHECK-NEXT: [[E:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NEXT: [[Splat3:%.*]] = insertelement <1 x i32> poison, i32 [[E]], i64 0
// CHECK-NEXT: [[F:%.*]] = extractelement <1 x i32> [[Splat3]], i32 0
// CHECK-NZ-NEXT: [[G:%.*]] = icmp ne i32 [[F]], 0
// CHECK-TR-NEXT: [[G:%.*]] = trunc i32 [[F]] to i1
// CHECK-NEXT: br label %lor.end

// CHECK: lor.end:
// CHECK-NEXT: [[H:%.*]] = phi i1 [ true, %entry ], [ [[G]], %lor.rhs ]
// CHECK-NEXT: [[J:%.*]] = zext i1 %9 to i32
// CHECK-NEXT: store i32 [[J]], ptr [[XAddr]], align 4
// CHECK-NEXT: [[I:%.*]] = load i32, ptr [[XAddr]], align 4
// CHECK-NZ-NEXT: [[LoadV:%.*]] = icmp ne i32 [[I]], 0
// CHECK-TR-NEXT: [[LoadV:%.*]] = trunc i32 [[I]] to i1
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
// CHECK-NZ-NEXT: [[LV:%.*]] = icmp ne <2 x i32> [[Z]], zeroinitializer
// CHECK-TR-NEXT: [[LV:%.*]] = trunc <2 x i32> [[Z]] to <2 x i1>
// CHECK-NEXT: [[A:%.*]] = zext <2 x i1> [[LV]] to <2 x i32>
// CHECK-NEXT: store <2 x i32> [[A]], ptr [[X]], align 8
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NZ-NEXT: [[LV1:%.*]] = icmp ne i32 [[B]], 0
// CHECK-TR-NEXT: [[LV1:%.*]] = trunc i32 [[B]] to i1
// CHECK-NEXT: [[D:%.*]] = zext i1 [[LV1]] to i32
// CHECK-NEXT: [[C:%.*]] = getelementptr <2 x i32>, ptr [[X]], i32 0, i32 1
// CHECK-NEXT: store i32 [[D]], ptr [[C]], align 4
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
// CHECK-NZ-NEXT: [[LV:%.*]] = icmp ne <2 x i32> [[Z]], zeroinitializer
// CHECK-TR-NEXT: [[LV:%.*]] = trunc <2 x i32> [[Z]] to <2 x i1>
// CHECK-NEXT: [[B:%.*]] = zext <2 x i1> [[LV]] to <2 x i32>
// CHECK-NEXT: [[V1:%.*]] = extractelement <2 x i32> [[B]], i32 0
// CHECK-NEXT: store i32 [[V1]], ptr [[X]], align 4
// CHECK-NEXT: [[V2:%.*]] = extractelement <2 x i32> [[B]], i32 1
// CHECK-NEXT: [[X2:%.*]] = getelementptr <2 x i32>, ptr [[X]], i32 0, i32 1
// CHECK-NEXT: store i32 [[V2]], ptr [[X2]], align 4
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
// CHECK-NZ-NEXT: [[LV:%.*]] = icmp ne <4 x i32> [[Z]], zeroinitializer
// CHECK-TR-NEXT: [[LV:%.*]] = trunc <4 x i32> [[Vec2]] to <4 x i1>
// CHECK-NEXT: [[A:%.*]] = zext <4 x i1> [[LV]] to <4 x i32>
// CHECK-NEXT: store <4 x i32> [[A]], ptr [[X]], align 16
// CHECK-NEXT: [[B:%.*]] = load <4 x i32>, ptr [[X]], align 16
// CHECK-NEXT: [[C:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> poison, <2 x i32> <i32 2, i32 3>
// CHECK-NZ-NEXT: [[LV1:%.*]] = icmp ne <2 x i32> [[C]], zeroinitializer
// CHECK-TR-NEXT: [[LV1:%.*]] = trunc <2 x i32> [[C]] to <2 x i1>
// CHECK-NEXT: ret <2 x i1> [[LV1]]
bool2 AccessBools() {
  bool4 X = true.xxxx;
  return X.zw;
}

// CHECK-LABEL: define hidden void {{.*}}BoolSizeMismatch{{.*}}
// CHECK: [[B:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT: [[Tmp:%.*]] = alloca <1 x i32>, align 4
// CHECK-NEXT: store <4 x i32> splat (i32 1), ptr [[B]], align 16
// CHECK-NEXT: store <1 x i32> zeroinitializer, ptr [[Tmp]], align 4
// CHECK-NEXT: [[L0:%.*]] = load <1 x i32>, ptr [[Tmp]], align 4
// CHECK-NEXT: [[L1:%.*]] = shufflevector <1 x i32> [[L0]], <1 x i32> poison, <3 x i32> zeroinitializer
// CHECK-NZ-NEXT: [[TruncV:%.*]] = icmp ne <3 x i32> [[L1]], zeroinitializer
// CHECK-TR-NEXT: [[TruncV:%.*]] = trunc <3 x i32> [[Vec2]] to <3 x i1>
// CHECK-NEXT: [[L2:%.*]] = zext <3 x i1> [[TruncV]] to <3 x i32>
// CHECK-NEXT: [[V1:%.*]] = extractelement <3 x i32> [[L2]], i32 0
// CHECK-NEXT: store i32 [[V1]], ptr %B, align 4
// CHECK-NEXT: [[V2:%.*]] = extractelement <3 x i32> [[L2]], i32 1
// CHECK-NEXT: [[B2:%.*]] = getelementptr <4 x i32>, ptr %B, i32 0, i32 1
// CHECK-NEXT: store i32 [[V2]], ptr [[B2]], align 4
// CHECK-NEXT: [[V3:%.*]] = extractelement <3 x i32> [[L2]], i32 2
// CHECK-NEXT: [[B3:%.*]] = getelementptr <4 x i32>, ptr %B, i32 0, i32 2
void BoolSizeMismatch() {
  bool4 B = {true,true,true,true};
  B.xyz = false.xxx;
}
