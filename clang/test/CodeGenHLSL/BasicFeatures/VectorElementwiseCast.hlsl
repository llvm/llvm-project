// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=column-major -o - %s | FileCheck %s --check-prefixes=CHECK,COL-CHECK
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=row-major -o - %s | FileCheck %s --check-prefixes=CHECK,ROW-CHECK

// vector flat cast from array
// CHECK-LABEL: define void {{.*}}call2
// CHECK: [[A:%.*]] = alloca [2 x [1 x i32]], align 4
// CHECK-NEXT: [[B:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x [1 x i32]], align 4
// CHECK-NEXT: [[Tmp2:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x [1 x i32]], ptr [[Tmp]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x [1 x i32]], ptr [[Tmp]], i32 0, i32 1, i32 0
// CHECK-NEXT: [[C:%.*]] = load <2 x i32>, ptr [[Tmp2]], align 8
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: [[D:%.*]] = insertelement <2 x i32> [[C]], i32 [[L]], i64 0
// CHECK-NEXT: [[L2:%.*]] = load i32, ptr [[G2]], align 4
// CHECK-NEXT: [[E:%.*]] = insertelement <2 x i32> [[D]], i32 [[L2]], i64 1
// CHECK-NEXT: store <2 x i32> [[E]], ptr [[B]], align 8
export void call2() {
  int A[2][1] = {{1},{2}};
  int2 B = (int2)A;
}

struct S {
  int X;
  float Y;
};

// vector flat cast from struct
// CHECK-LABEL: define void {{.*}}call3
// CHECK: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[A:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[Tmp2:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[s]], ptr align 1 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[s]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[B:%.*]] = load <2 x i32>, ptr [[Tmp2]], align 8
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: [[C:%.*]] = insertelement <2 x i32> [[B]], i32 [[L]], i64 0
// CHECK-NEXT: [[L2:%.*]] = load float, ptr [[G2]], align 4
// CHECK-NEXT: [[D:%.*]] = fptosi float [[L2]] to i32
// CHECK-NEXT: [[E:%.*]] = insertelement <2 x i32> [[C]], i32 [[D]], i64 1
// CHECK-NEXT: store <2 x i32> [[E]], ptr [[A]], align 8
export void call3() {
  S s = {1, 2.0};
  int2 A = (int2)s;
}

// truncate array to scalar
// CHECK-LABEL: define void {{.*}}call4
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[B:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[B]], align 4
export void call4() {
 int A[2] = {1,2};
 int B = (int)A;
}

// truncate struct to scalar
// CHECK-LABEL: define void {{.*}}call5
// CHECK: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[A:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[s]], ptr align 1 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[s]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[A]], align 4
export void call5() {
 S s = {1, 2.0};
 int A = (int)s;
}

struct BFields {
  double D;
  int E: 15;
  int : 8;
  float F;
};

struct Derived : BFields {
  int G;
};

// vector flat cast from derived struct with bitfield
// CHECK-LABEL: call6
// CHECK: [[A:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Derived, align 1
// CHECK-NEXT: [[FlatTmp:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 %D, i32 19, i1 false)
// CHECK-NEXT: [[Gep:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[E:%.*]] = getelementptr inbounds nuw %struct.BFields, ptr [[Gep]], i32 0, i32 1
// CHECK-NEXT: [[Gep1:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[Gep2:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0, i32 2
// CHECK-NEXT: [[Gep3:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[Z:%.*]] = load <4 x i32>, ptr [[FlatTmp]], align 16
// CHECK-NEXT: [[Y:%.*]] = load double, ptr [[Gep1]], align 8
// CHECK-NEXT: [[C:%.*]] = fptosi double [[Y]] to i32
// CHECK-NEXT: [[X:%.*]] = insertelement <4 x i32> [[Z]], i32 [[C]], i64 0
// CHECK-NEXT: [[BFL:%.*]] = load i24, ptr [[E]], align 1
// CHECK-NEXT: [[BFShl:%.*]] = shl i24 [[BFL]], 9
// CHECK-NEXT: [[BFAshr:%.*]] = ashr i24 [[BFShl]], 9
// CHECK-NEXT: [[BFC:%.*]] = sext i24 [[BFAshr]] to i32
// CHECK-NEXT: [[W:%.*]] = insertelement <4 x i32> [[X]], i32 [[BFC]], i64 1
// CHECK-NEXT: [[V:%.*]] = load float, ptr [[Gep2]], align 4
// CHECK-NEXT: [[C4:%.*]] = fptosi float [[V]] to i32
// CHECK-NEXT: [[U:%.*]] = insertelement <4 x i32> [[W]], i32 [[C4]], i64 2
// CHECK-NEXT: [[T:%.*]] = load i32, ptr [[Gep3]], align 4
// CHECK-NEXT: [[S:%.*]] = insertelement <4 x i32> [[U]], i32 [[T]], i64 3
// CHECK-NEXT: store <4 x i32> [[S]], ptr [[A]], align 16
// CHECK-NEXT: ret void
export void call6(Derived D) {
  int4 A = (int4)D;
}

// vector flat cast from matrix of same size (float)
// CHECK-LABEL: call7
// CHECK:    [[M_ADDR:%.*]] = alloca [2 x <2 x float>], align 4
// CHECK-NEXT:    [[V:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[HLSL_EWCAST_SRC:%.*]] = alloca [2 x <2 x float>], align 4
// CHECK-NEXT:    [[FLATCAST_TMP:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    store <4 x float> %M, ptr [[M_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x float>, ptr [[M_ADDR]], align 4
// CHECK-NEXT:    store <4 x float> [[TMP0]], ptr [[HLSL_EWCAST_SRC]], align 4
// CHECK-NEXT:    [[MATRIX_GEP:%.*]] = getelementptr inbounds <4 x float>, ptr [[HLSL_EWCAST_SRC]], i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, ptr [[FLATCAST_TMP]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load <4 x float>, ptr [[MATRIX_GEP]], align 4
// CHECK-NEXT:    [[MATRIXEXT:%.*]] = extractelement <4 x float> [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x float> [[TMP1]], float [[MATRIXEXT]], i64 0
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x float>, ptr [[MATRIX_GEP]], align 4
// COL-CHECK-NEXT:    [[MATRIXEXT1:%.*]] = extractelement <4 x float> [[TMP4]], i32 2
// ROW-CHECK-NEXT:    [[MATRIXEXT1:%.*]] = extractelement <4 x float> [[TMP4]], i32 1
// CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x float> [[TMP3]], float [[MATRIXEXT1]], i64 1
// CHECK-NEXT:    [[TMP6:%.*]] = load <4 x float>, ptr [[MATRIX_GEP]], align 4
// COL-CHECK-NEXT:    [[MATRIXEXT2:%.*]] = extractelement <4 x float> [[TMP6]], i32 1
// ROW-CHECK-NEXT:    [[MATRIXEXT2:%.*]] = extractelement <4 x float> [[TMP6]], i32 2
// CHECK-NEXT:    [[TMP7:%.*]] = insertelement <4 x float> [[TMP5]], float [[MATRIXEXT2]], i64 2
// CHECK-NEXT:    [[TMP8:%.*]] = load <4 x float>, ptr [[MATRIX_GEP]], align 4
// CHECK-NEXT:    [[MATRIXEXT3:%.*]] = extractelement <4 x float> [[TMP8]], i32 3
// CHECK-NEXT:    [[TMP9:%.*]] = insertelement <4 x float> [[TMP7]], float [[MATRIXEXT3]], i64 3
// CHECK-NEXT:    store <4 x float> [[TMP9]], ptr [[V]], align 16
// CHECK-NEXT:    ret void
export void call7(float2x2 M) {
    float4 V = (float4)M;
}

// vector flat cast from matrix of same size (int)
// CHECK-LABEL: call8
// COL-CHECK:    [[M_ADDR:%.*]] = alloca [1 x <3 x i32>], align 4
// ROW-CHECK:    [[M_ADDR:%.*]] = alloca [3 x <1 x i32>], align 4
// CHECK-NEXT:    [[V:%.*]] = alloca <3 x i32>, align 16
// COL-CHECK-NEXT:    [[HLSL_EWCAST_SRC:%.*]] = alloca [1 x <3 x i32>], align 4
// ROW-CHECK-NEXT:    [[HLSL_EWCAST_SRC:%.*]] = alloca [3 x <1 x i32>], align 4
// CHECK-NEXT:    [[FLATCAST_TMP:%.*]] = alloca <3 x i32>, align 16
// CHECK-NEXT:    store <3 x i32> %M, ptr [[M_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load <3 x i32>, ptr [[M_ADDR]], align 4
// CHECK-NEXT:    store <3 x i32> [[TMP0]], ptr [[HLSL_EWCAST_SRC]], align 4
// CHECK-NEXT:    [[MATRIX_GEP:%.*]] = getelementptr inbounds <3 x i32>, ptr [[HLSL_EWCAST_SRC]], i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load <3 x i32>, ptr [[FLATCAST_TMP]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load <3 x i32>, ptr [[MATRIX_GEP]], align 4
// CHECK-NEXT:    [[MATRIXEXT:%.*]] = extractelement <3 x i32> [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = insertelement <3 x i32> [[TMP1]], i32 [[MATRIXEXT]], i64 0
// CHECK-NEXT:    [[TMP4:%.*]] = load <3 x i32>, ptr [[MATRIX_GEP]], align 4
// CHECK-NEXT:    [[MATRIXEXT1:%.*]] = extractelement <3 x i32> [[TMP4]], i32 1
// CHECK-NEXT:    [[TMP5:%.*]] = insertelement <3 x i32> [[TMP3]], i32 [[MATRIXEXT1]], i64 1
// CHECK-NEXT:    [[TMP6:%.*]] = load <3 x i32>, ptr [[MATRIX_GEP]], align 4
// CHECK-NEXT:    [[MATRIXEXT2:%.*]] = extractelement <3 x i32> [[TMP6]], i32 2
// CHECK-NEXT:    [[TMP7:%.*]] = insertelement <3 x i32> [[TMP5]], i32 [[MATRIXEXT2]], i64 2
// CHECK-NEXT:    store <3 x i32> [[TMP7]], ptr [[V]], align 16
// CHECK-NEXT:    ret void
export void call8(int3x1 M) {
    int3 V = (int3)M;
}

