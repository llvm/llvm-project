// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// array splat
// CHECK-LABEL: define void {{.*}}call4
// CHECK: [[B:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[B]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 1
// CHECK-NEXT: store i32 3, ptr [[G1]], align 4
// CHECK-NEXT: store i32 3, ptr [[G2]], align 4
export void call4() {
  int B[2] = {1,2};
  B = (int[2])3;
}

// splat from vector of length 1
// CHECK-LABEL: define void {{.*}}call8
// CHECK: [[A:%.*]] = alloca <1 x i32>, align 4
// CHECK-NEXT: [[B:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: store <1 x i32> splat (i32 1), ptr [[A]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[B]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: [[L:%.*]] = load <1 x i32>, ptr [[A]], align 4
// CHECK-NEXT: [[VL:%.*]] = extractelement <1 x i32> [[L]], i32 0
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 1
// CHECK-NEXT: store i32 [[VL]], ptr [[G1]], align 4
// CHECK-NEXT: store i32 [[VL]], ptr [[G2]], align 4
export void call8() {
  int1 A = {1};
  int B[2] = {1,2};
  B = (int[2])A;
}

// vector splat from vector of length 1
// CHECK-LABEL: define void {{.*}}call1
// CHECK: [[B:%.*]] = alloca <1 x float>, align 4
// CHECK-NEXT: [[A:%.*]] = alloca <4 x i32>, align 4
// CHECK-NEXT: store <1 x float> splat (float 1.000000e+00), ptr [[B]], align 4
// CHECK-NEXT: [[L:%.*]] = load <1 x float>, ptr [[B]], align 4
// CHECK-NEXT: [[VL:%.*]] = extractelement <1 x float> [[L]], i32 0
// CHECK-NEXT: [[C:%.*]] = fptosi float [[VL]] to i32
// CHECK-NEXT: [[SI:%.*]] = insertelement <4 x i32> poison, i32 [[C]], i64 0
// CHECK-NEXT: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK-NEXT: store <4 x i32> [[S]], ptr [[A]], align 4
export void call1() {
  float1 B = {1.0};
  int4 A = (int4)B;
}

struct S {
  int X;
  float Y;
};

// struct splats
// CHECK-LABEL: define void @_Z5call3i(
// CHECK-SAME: i32 noundef [[A:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[A_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[S:%.*]] = alloca [[STRUCT_S:%.*]], align 1
// CHECK-NEXT:    [[REF_TMP:%.*]] = alloca [[STRUCT_S]], align 1
// CHECK-NEXT:    [[REF_TMP3:%.*]] = alloca [[STRUCT_S]], align 1
// CHECK-NEXT:    store i32 [[A]], ptr [[A_ADDR]], align 4
// CHECK-NEXT:    [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[S]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP]], i32 0, i32 0
// CHECK-NEXT:    [[GEP1:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP]], i32 0, i32 1
// CHECK-NEXT:    store i32 [[TMP0]], ptr [[GEP]], align 4
// CHECK-NEXT:    [[CONV:%.*]] = sitofp i32 [[TMP0]] to float
// CHECK-NEXT:    store float [[CONV]], ptr [[GEP1]], align 4
// CHECK-NEXT:    [[X2:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[REF_TMP]], i32 0, i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[X2]], align 1
// CHECK-NEXT:    store i32 [[TMP1]], ptr [[X]], align 1
// CHECK-NEXT:    [[Y:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[S]], i32 0, i32 1
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK-NEXT:    [[GEP4:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP3]], i32 0, i32 0
// CHECK-NEXT:    [[GEP5:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP3]], i32 0, i32 1
// CHECK-NEXT:    store i32 [[TMP2]], ptr [[GEP4]], align 4
// CHECK-NEXT:    [[CONV6:%.*]] = sitofp i32 [[TMP2]] to float
// CHECK-NEXT:    store float [[CONV6]], ptr [[GEP5]], align 4
// CHECK-NEXT:    [[Y7:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[REF_TMP3]], i32 0, i32 1
// CHECK-NEXT:    [[TMP3:%.*]] = load float, ptr [[Y7]], align 1
// CHECK-NEXT:    store float [[TMP3]], ptr [[Y]], align 1
// CHECK-NEXT:    ret void
//
export void call3(int A) {
  S s = (S)A;
}

// struct splat from vector of length 1
// CHECK-LABEL: define void @_Z5call5v(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[A:%.*]] = alloca <1 x i32>, align 4
// CHECK-NEXT:    [[S:%.*]] = alloca [[STRUCT_S:%.*]], align 1
// CHECK-NEXT:    [[REF_TMP:%.*]] = alloca [[STRUCT_S]], align 1
// CHECK-NEXT:    [[REF_TMP3:%.*]] = alloca [[STRUCT_S]], align 1
// CHECK-NEXT:    store <1 x i32> splat (i32 1), ptr [[A]], align 4
// CHECK-NEXT:    [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[S]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load <1 x i32>, ptr [[A]], align 4
// CHECK-NEXT:    [[CAST_VTRUNC:%.*]] = extractelement <1 x i32> [[TMP0]], i32 0
// CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP]], i32 0, i32 0
// CHECK-NEXT:    [[GEP1:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP]], i32 0, i32 1
// CHECK-NEXT:    store i32 [[CAST_VTRUNC]], ptr [[GEP]], align 4
// CHECK-NEXT:    [[CONV:%.*]] = sitofp i32 [[CAST_VTRUNC]] to float
// CHECK-NEXT:    store float [[CONV]], ptr [[GEP1]], align 4
// CHECK-NEXT:    [[X2:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[REF_TMP]], i32 0, i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[X2]], align 1
// CHECK-NEXT:    store i32 [[TMP1]], ptr [[X]], align 1
// CHECK-NEXT:    [[Y:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[S]], i32 0, i32 1
// CHECK-NEXT:    [[TMP2:%.*]] = load <1 x i32>, ptr [[A]], align 4
// CHECK-NEXT:    [[CAST_VTRUNC4:%.*]] = extractelement <1 x i32> [[TMP2]], i32 0
// CHECK-NEXT:    [[GEP5:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP3]], i32 0, i32 0
// CHECK-NEXT:    [[GEP6:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP3]], i32 0, i32 1
// CHECK-NEXT:    store i32 [[CAST_VTRUNC4]], ptr [[GEP5]], align 4
// CHECK-NEXT:    [[CONV7:%.*]] = sitofp i32 [[CAST_VTRUNC4]] to float
// CHECK-NEXT:    store float [[CONV7]], ptr [[GEP6]], align 4
// CHECK-NEXT:    [[Y8:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[REF_TMP3]], i32 0, i32 1
// CHECK-NEXT:    [[TMP3:%.*]] = load float, ptr [[Y8]], align 1
// CHECK-NEXT:    store float [[TMP3]], ptr [[Y]], align 1
// CHECK-NEXT:    ret void
//
export void call5() {
  int1 A = {1};
  S s = (S)A;
}

// vector splat from 1x1 matrix
// CHECK-LABEL: define void {{.*}}call9
// CHECK: [[M:%.*]] = alloca [1 x <1 x float>], align 4
// CHECK-NEXT: [[A:%.*]] = alloca <4 x i32>, align 4
// CHECK-NEXT: store <1 x float> {{.*}}, ptr [[M]], align 4
// CHECK-NEXT: [[L:%.*]] = load <1 x float>, ptr [[M]], align 4
// CHECK-NEXT: [[ML:%.*]] = extractelement <1 x float> [[L]], i32 0
// CHECK-NEXT: [[C:%.*]] = fptosi float [[ML]] to i32
// CHECK-NEXT: [[SI:%.*]] = insertelement <4 x i32> poison, i32 [[C]], i64 0
// CHECK-NEXT: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK-NEXT: store <4 x i32> [[S]], ptr [[A]], align 4
export void call9() {
  float1x1 M = {1.0};
  int4 A = (int4)M;
}

// struct splat from 1x1 matrix
// CHECK-LABEL: define void @_Z6call10v(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[M:%.*]] = alloca [1 x <1 x i32>], align 4
// CHECK-NEXT:    [[S:%.*]] = alloca [[STRUCT_S:%.*]], align 1
// CHECK-NEXT:    [[REF_TMP:%.*]] = alloca [[STRUCT_S]], align 1
// CHECK-NEXT:    [[REF_TMP3:%.*]] = alloca [[STRUCT_S]], align 1
// CHECK-NEXT:    store <1 x i32> splat (i32 1), ptr [[M]], align 4
// CHECK-NEXT:    [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[S]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load <1 x i32>, ptr [[M]], align 4
// CHECK-NEXT:    [[CAST_MTRUNC:%.*]] = extractelement <1 x i32> [[TMP0]], i32 0
// CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP]], i32 0, i32 0
// CHECK-NEXT:    [[GEP1:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP]], i32 0, i32 1
// CHECK-NEXT:    store i32 [[CAST_MTRUNC]], ptr [[GEP]], align 4
// CHECK-NEXT:    [[CONV:%.*]] = sitofp i32 [[CAST_MTRUNC]] to float
// CHECK-NEXT:    store float [[CONV]], ptr [[GEP1]], align 4
// CHECK-NEXT:    [[X2:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[REF_TMP]], i32 0, i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[X2]], align 1
// CHECK-NEXT:    store i32 [[TMP1]], ptr [[X]], align 1
// CHECK-NEXT:    [[Y:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[S]], i32 0, i32 1
// CHECK-NEXT:    [[TMP2:%.*]] = load <1 x i32>, ptr [[M]], align 4
// CHECK-NEXT:    [[CAST_MTRUNC4:%.*]] = extractelement <1 x i32> [[TMP2]], i32 0
// CHECK-NEXT:    [[GEP5:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP3]], i32 0, i32 0
// CHECK-NEXT:    [[GEP6:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[REF_TMP3]], i32 0, i32 1
// CHECK-NEXT:    store i32 [[CAST_MTRUNC4]], ptr [[GEP5]], align 4
// CHECK-NEXT:    [[CONV7:%.*]] = sitofp i32 [[CAST_MTRUNC4]] to float
// CHECK-NEXT:    store float [[CONV7]], ptr [[GEP6]], align 4
// CHECK-NEXT:    [[Y8:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[REF_TMP3]], i32 0, i32 1
// CHECK-NEXT:    [[TMP3:%.*]] = load float, ptr [[Y8]], align 1
// CHECK-NEXT:    store float [[TMP3]], ptr [[Y]], align 1
// CHECK-NEXT:    ret void
//
export void call10() {
  int1x1 M = {1};
  S s = (S)M;
}

struct BFields {
  double DF;
  int E: 15;
  int : 8;
  float F;
};

struct Derived : BFields {
  int G;
};

// derived struct with bitfields splat from scalar
// CHECK-LABEL: define void @_Z5call6i(
// CHECK-SAME: i32 noundef [[A:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[A_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[D:%.*]] = alloca [[STRUCT_DERIVED:%.*]], align 1
// CHECK-NEXT:    [[REF_TMP:%.*]] = alloca [[STRUCT_DERIVED]], align 1
// CHECK-NEXT:    [[REF_TMP7:%.*]] = alloca [[STRUCT_DERIVED]], align 1
// CHECK-NEXT:    [[REF_TMP25:%.*]] = alloca [[STRUCT_DERIVED]], align 1
// CHECK-NEXT:    [[REF_TMP38:%.*]] = alloca [[STRUCT_DERIVED]], align 1
// CHECK-NEXT:    store i32 [[A]], ptr [[A_ADDR]], align 4
// CHECK-NEXT:    [[DF:%.*]] = getelementptr inbounds nuw [[STRUCT_BFIELDS:%.*]], ptr [[D]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP]], i32 0, i32 0
// CHECK-NEXT:    [[E:%.*]] = getelementptr inbounds nuw [[STRUCT_BFIELDS]], ptr [[GEP]], i32 0, i32 1
// CHECK-NEXT:    [[GEP1:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP]], i32 0, i32 0, i32 0
// CHECK-NEXT:    [[GEP2:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP]], i32 0, i32 0, i32 2
// CHECK-NEXT:    [[GEP3:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP]], i32 0, i32 1
// CHECK-NEXT:    [[CONV:%.*]] = sitofp i32 [[TMP0]] to double
// CHECK-NEXT:    store double [[CONV]], ptr [[GEP1]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 [[TMP0]] to i24
// CHECK-NEXT:    [[BF_LOAD:%.*]] = load i24, ptr [[E]], align 1
// CHECK-NEXT:    [[BF_VALUE:%.*]] = and i24 [[TMP1]], 32767
// CHECK-NEXT:    [[BF_CLEAR:%.*]] = and i24 [[BF_LOAD]], -32768
// CHECK-NEXT:    [[BF_SET:%.*]] = or i24 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK-NEXT:    store i24 [[BF_SET]], ptr [[E]], align 1
// CHECK-NEXT:    [[CONV4:%.*]] = sitofp i32 [[TMP0]] to float
// CHECK-NEXT:    store float [[CONV4]], ptr [[GEP2]], align 4
// CHECK-NEXT:    store i32 [[TMP0]], ptr [[GEP3]], align 4
// CHECK-NEXT:    [[DF5:%.*]] = getelementptr inbounds nuw [[STRUCT_BFIELDS]], ptr [[REF_TMP]], i32 0, i32 0
// CHECK-NEXT:    [[TMP2:%.*]] = load double, ptr [[DF5]], align 1
// CHECK-NEXT:    store double [[TMP2]], ptr [[DF]], align 1
// CHECK-NEXT:    [[E6:%.*]] = getelementptr inbounds nuw [[STRUCT_BFIELDS]], ptr [[D]], i32 0, i32 1
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK-NEXT:    [[GEP8:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP7]], i32 0, i32 0
// CHECK-NEXT:    [[E9:%.*]] = getelementptr inbounds nuw [[STRUCT_BFIELDS]], ptr [[GEP8]], i32 0, i32 1
// CHECK-NEXT:    [[GEP10:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP7]], i32 0, i32 0, i32 0
// CHECK-NEXT:    [[GEP11:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP7]], i32 0, i32 0, i32 2
// CHECK-NEXT:    [[GEP12:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP7]], i32 0, i32 1
// CHECK-NEXT:    [[CONV13:%.*]] = sitofp i32 [[TMP3]] to double
// CHECK-NEXT:    store double [[CONV13]], ptr [[GEP10]], align 8
// CHECK-NEXT:    [[TMP4:%.*]] = trunc i32 [[TMP3]] to i24
// CHECK-NEXT:    [[BF_LOAD14:%.*]] = load i24, ptr [[E9]], align 1
// CHECK-NEXT:    [[BF_VALUE15:%.*]] = and i24 [[TMP4]], 32767
// CHECK-NEXT:    [[BF_CLEAR16:%.*]] = and i24 [[BF_LOAD14]], -32768
// CHECK-NEXT:    [[BF_SET17:%.*]] = or i24 [[BF_CLEAR16]], [[BF_VALUE15]]
// CHECK-NEXT:    store i24 [[BF_SET17]], ptr [[E9]], align 1
// CHECK-NEXT:    [[CONV18:%.*]] = sitofp i32 [[TMP3]] to float
// CHECK-NEXT:    store float [[CONV18]], ptr [[GEP11]], align 4
// CHECK-NEXT:    store i32 [[TMP3]], ptr [[GEP12]], align 4
// CHECK-NEXT:    [[E19:%.*]] = getelementptr inbounds nuw [[STRUCT_BFIELDS]], ptr [[REF_TMP7]], i32 0, i32 1
// CHECK-NEXT:    [[BF_LOAD20:%.*]] = load i24, ptr [[E19]], align 1
// CHECK-NEXT:    [[BF_SHL:%.*]] = shl i24 [[BF_LOAD20]], 9
// CHECK-NEXT:    [[BF_ASHR:%.*]] = ashr i24 [[BF_SHL]], 9
// CHECK-NEXT:    [[BF_CAST:%.*]] = sext i24 [[BF_ASHR]] to i32
// CHECK-NEXT:    [[TMP5:%.*]] = trunc i32 [[BF_CAST]] to i24
// CHECK-NEXT:    [[BF_LOAD21:%.*]] = load i24, ptr [[E6]], align 1
// CHECK-NEXT:    [[BF_VALUE22:%.*]] = and i24 [[TMP5]], 32767
// CHECK-NEXT:    [[BF_CLEAR23:%.*]] = and i24 [[BF_LOAD21]], -32768
// CHECK-NEXT:    [[BF_SET24:%.*]] = or i24 [[BF_CLEAR23]], [[BF_VALUE22]]
// CHECK-NEXT:    store i24 [[BF_SET24]], ptr [[E6]], align 1
// CHECK-NEXT:    [[F:%.*]] = getelementptr inbounds nuw [[STRUCT_BFIELDS]], ptr [[D]], i32 0, i32 2
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK-NEXT:    [[GEP26:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP25]], i32 0, i32 0
// CHECK-NEXT:    [[E27:%.*]] = getelementptr inbounds nuw [[STRUCT_BFIELDS]], ptr [[GEP26]], i32 0, i32 1
// CHECK-NEXT:    [[GEP28:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP25]], i32 0, i32 0, i32 0
// CHECK-NEXT:    [[GEP29:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP25]], i32 0, i32 0, i32 2
// CHECK-NEXT:    [[GEP30:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP25]], i32 0, i32 1
// CHECK-NEXT:    [[CONV31:%.*]] = sitofp i32 [[TMP6]] to double
// CHECK-NEXT:    store double [[CONV31]], ptr [[GEP28]], align 8
// CHECK-NEXT:    [[TMP7:%.*]] = trunc i32 [[TMP6]] to i24
// CHECK-NEXT:    [[BF_LOAD32:%.*]] = load i24, ptr [[E27]], align 1
// CHECK-NEXT:    [[BF_VALUE33:%.*]] = and i24 [[TMP7]], 32767
// CHECK-NEXT:    [[BF_CLEAR34:%.*]] = and i24 [[BF_LOAD32]], -32768
// CHECK-NEXT:    [[BF_SET35:%.*]] = or i24 [[BF_CLEAR34]], [[BF_VALUE33]]
// CHECK-NEXT:    store i24 [[BF_SET35]], ptr [[E27]], align 1
// CHECK-NEXT:    [[CONV36:%.*]] = sitofp i32 [[TMP6]] to float
// CHECK-NEXT:    store float [[CONV36]], ptr [[GEP29]], align 4
// CHECK-NEXT:    store i32 [[TMP6]], ptr [[GEP30]], align 4
// CHECK-NEXT:    [[F37:%.*]] = getelementptr inbounds nuw [[STRUCT_BFIELDS]], ptr [[REF_TMP25]], i32 0, i32 2
// CHECK-NEXT:    [[TMP8:%.*]] = load float, ptr [[F37]], align 1
// CHECK-NEXT:    store float [[TMP8]], ptr [[F]], align 1
// CHECK-NEXT:    [[G:%.*]] = getelementptr inbounds nuw [[STRUCT_DERIVED]], ptr [[D]], i32 0, i32 1
// CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK-NEXT:    [[GEP39:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP38]], i32 0, i32 0
// CHECK-NEXT:    [[E40:%.*]] = getelementptr inbounds nuw [[STRUCT_BFIELDS]], ptr [[GEP39]], i32 0, i32 1
// CHECK-NEXT:    [[GEP41:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP38]], i32 0, i32 0, i32 0
// CHECK-NEXT:    [[GEP42:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP38]], i32 0, i32 0, i32 2
// CHECK-NEXT:    [[GEP43:%.*]] = getelementptr inbounds [[STRUCT_DERIVED]], ptr [[REF_TMP38]], i32 0, i32 1
// CHECK-NEXT:    [[CONV44:%.*]] = sitofp i32 [[TMP9]] to double
// CHECK-NEXT:    store double [[CONV44]], ptr [[GEP41]], align 8
// CHECK-NEXT:    [[TMP10:%.*]] = trunc i32 [[TMP9]] to i24
// CHECK-NEXT:    [[BF_LOAD45:%.*]] = load i24, ptr [[E40]], align 1
// CHECK-NEXT:    [[BF_VALUE46:%.*]] = and i24 [[TMP10]], 32767
// CHECK-NEXT:    [[BF_CLEAR47:%.*]] = and i24 [[BF_LOAD45]], -32768
// CHECK-NEXT:    [[BF_SET48:%.*]] = or i24 [[BF_CLEAR47]], [[BF_VALUE46]]
// CHECK-NEXT:    store i24 [[BF_SET48]], ptr [[E40]], align 1
// CHECK-NEXT:    [[CONV49:%.*]] = sitofp i32 [[TMP9]] to float
// CHECK-NEXT:    store float [[CONV49]], ptr [[GEP42]], align 4
// CHECK-NEXT:    store i32 [[TMP9]], ptr [[GEP43]], align 4
// CHECK-NEXT:    [[G50:%.*]] = getelementptr inbounds nuw [[STRUCT_DERIVED]], ptr [[REF_TMP38]], i32 0, i32 1
// CHECK-NEXT:    [[TMP11:%.*]] = load i32, ptr [[G50]], align 1
// CHECK-NEXT:    store i32 [[TMP11]], ptr [[G]], align 1
// CHECK-NEXT:    ret void
//
export void call6(int A) {
  Derived D = (Derived)A;
}
